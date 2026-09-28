from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

PLUGIN_ID = "astrbot_plugin_natural_companion"

DEFAULT_CONFIG: dict[str, Any] = {
    "min_delay_minutes": 20,
    "max_delay_minutes": 240,
    "min_gap_minutes": 45,
    "max_proactive_per_24h": 2,
    "quiet_hours_enabled": True,
    "quiet_start": "00:00",
    "quiet_end": "07:00",
    "state_extract_debounce_seconds": 15,
    "opportunity_threshold": 0.55,
    "natural_greeting_threshold": 0.72,
    "natural_greeting_min_delay_minutes": 60,
    "test_delay_seconds": 10,
    "allow_natural_greeting": True,
    "max_history_chars": 7000,
    "max_persona_chars": 3500,
    "max_message_chars": 500,
    "debug_logging": False,
    "phone_bridge_enabled": False,
    "phone_bridge_path": "",
    "phone_bridge_check_seconds": 5,
}

REASON_TYPES = {
    "unfinished_topic",
    "emotional_residue",
    "memory_cue",
    "scene_change",
    "natural_greeting",
    "mixed",
}


def now_ts() -> float:
    return time.time()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def truncate(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any, limit: int = 8, item_chars: int = 240) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("text") or item.get("summary") or item.get("topic") or ""
        text = truncate(item, item_chars)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    """Parse a JSON object from a model response, tolerating markdown fences."""
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return dict(parsed) if isinstance(parsed, Mapping) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            return dict(parsed) if isinstance(parsed, Mapping) else None
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def normalize_message(text: Any, max_chars: int = 500) -> str:
    message = str(text or "").strip()
    message = re.sub(r"^```(?:text|markdown)?\s*", "", message, flags=re.IGNORECASE)
    message = re.sub(r"\s*```$", "", message).strip()
    message = re.sub(r"\n{3,}", "\n\n", message)
    if not message:
        return ""

    # Keep proactive messages short without destroying normal Chinese punctuation.
    sentences = re.split(r"(?<=[。！？!?])\s*", message)
    if len(sentences) > 3:
        message = "".join(sentences[:3]).strip()
    return truncate(message, max_chars)


@dataclass
class MoodState:
    label: str = "平静"
    valence: float = 0.0
    arousal: float = 0.35
    note: str = ""

    @classmethod
    def from_dict(cls, value: Any) -> "MoodState":
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            label=truncate(value.get("label", "平静"), 40) or "平静",
            valence=clamp(_safe_float(value.get("valence"), 0.0), -1.0, 1.0),
            arousal=clamp(_safe_float(value.get("arousal"), 0.35), 0.0, 1.0),
            note=truncate(value.get("note", ""), 240),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "note": self.note,
        }


@dataclass
class CompanionState:
    version: int = 1
    enabled: bool = False
    bound_umo: str = ""
    last_user_message_at: float = 0.0
    last_proactive_message_at: float = 0.0
    last_interaction_summary: str = ""
    current_scene: str = ""
    mood: MoodState = field(default_factory=MoodState)
    unfinished_topics: list[str] = field(default_factory=list)
    memory_cues: list[str] = field(default_factory=list)
    relationship_tone: str = "普通"
    user_preferences: list[str] = field(default_factory=list)
    quiet_until: float = 0.0
    pending_opportunity: dict[str, Any] | None = None
    phone_binding_id: str = ""
    phone_event_ids: list[str] = field(default_factory=list)
    phone_cursor_ready: bool = False
    last_phone_message_at: float = 0.0
    recent_proactive_outcomes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Any) -> "CompanionState":
        if not isinstance(value, Mapping):
            return cls()
        pending = value.get("pending_opportunity")
        if not isinstance(pending, Mapping):
            pending = None
        outcomes = value.get("recent_proactive_outcomes")
        if not isinstance(outcomes, list):
            outcomes = []
        clean_outcomes: list[dict[str, Any]] = []
        for item in outcomes[-20:]:
            if isinstance(item, Mapping):
                clean_outcomes.append(dict(item))
        return cls(
            version=max(1, _safe_int(value.get("version"), 1)),
            enabled=bool(value.get("enabled", False)),
            bound_umo=truncate(value.get("bound_umo", ""), 400),
            last_user_message_at=_safe_float(value.get("last_user_message_at"), 0.0),
            last_proactive_message_at=_safe_float(
                value.get("last_proactive_message_at"), 0.0
            ),
            last_interaction_summary=truncate(
                value.get("last_interaction_summary", ""), 500
            ),
            current_scene=truncate(value.get("current_scene", ""), 500),
            mood=MoodState.from_dict(value.get("mood")),
            unfinished_topics=_string_list(value.get("unfinished_topics")),
            memory_cues=_string_list(value.get("memory_cues")),
            relationship_tone=truncate(value.get("relationship_tone", "普通"), 60)
            or "普通",
            user_preferences=_string_list(value.get("user_preferences")),
            quiet_until=_safe_float(value.get("quiet_until"), 0.0),
            pending_opportunity=dict(pending) if pending else None,
            phone_binding_id=truncate(value.get("phone_binding_id", ""), 160),
            phone_event_ids=_string_list(value.get("phone_event_ids"), 80, 160),
            phone_cursor_ready=bool(value.get("phone_cursor_ready", False)),
            last_phone_message_at=_safe_float(value.get("last_phone_message_at"), 0.0),
            recent_proactive_outcomes=clean_outcomes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "bound_umo": self.bound_umo,
            "last_user_message_at": self.last_user_message_at,
            "last_proactive_message_at": self.last_proactive_message_at,
            "last_interaction_summary": self.last_interaction_summary,
            "current_scene": self.current_scene,
            "mood": self.mood.to_dict(),
            "unfinished_topics": list(self.unfinished_topics),
            "memory_cues": list(self.memory_cues),
            "relationship_tone": self.relationship_tone,
            "user_preferences": list(self.user_preferences),
            "quiet_until": self.quiet_until,
            "pending_opportunity": self.pending_opportunity,
            "phone_binding_id": self.phone_binding_id,
            "phone_event_ids": list(self.phone_event_ids[-80:]),
            "phone_cursor_ready": self.phone_cursor_ready,
            "last_phone_message_at": self.last_phone_message_at,
            "recent_proactive_outcomes": list(self.recent_proactive_outcomes[-20:]),
        }

    def clear_plugin_memory(self) -> None:
        self.last_user_message_at = 0.0
        self.last_proactive_message_at = 0.0
        self.last_interaction_summary = ""
        self.current_scene = ""
        self.mood = MoodState()
        self.unfinished_topics.clear()
        self.memory_cues.clear()
        self.relationship_tone = "普通"
        self.user_preferences.clear()
        self.quiet_until = 0.0
        self.pending_opportunity = None
        self.phone_binding_id = ""
        self.phone_event_ids.clear()
        self.phone_cursor_ready = False
        self.last_phone_message_at = 0.0
        self.recent_proactive_outcomes.clear()


class StateStore:
    """Small JSON store used instead of AstrBot KV APIs for 4.5.7 compatibility."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> CompanionState:
        try:
            if not self.path.exists():
                return CompanionState()
            with self.path.open("r", encoding="utf-8") as handle:
                return CompanionState.from_dict(json.load(handle))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return CompanionState()

    def save(self, state: CompanionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)


def merge_extraction(state: CompanionState, payload: Mapping[str, Any] | None) -> CompanionState:
    """Apply only validated fields returned by the state-extraction model."""
    if not isinstance(payload, Mapping):
        return state

    mood_value = payload.get("mood")
    if isinstance(mood_value, Mapping):
        state.mood = MoodState.from_dict(mood_value)

    for field_name, max_chars in (
        ("current_scene", 500),
        ("relationship_tone", 60),
    ):
        if field_name in payload and payload[field_name] is not None:
            text = truncate(payload[field_name], max_chars)
            if text:
                setattr(state, field_name, text)

    for field_name in ("unfinished_topics", "memory_cues", "user_preferences"):
        if field_name in payload:
            setattr(state, field_name, _string_list(payload[field_name]))
    return state


def _parse_hhmm(value: Any, default: tuple[int, int]) -> int:
    try:
        hours, minutes = str(value).split(":", 1)
        hour = int(hours)
        minute = int(minutes)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
        return hour * 60 + minute
    except (ValueError, TypeError):
        return default[0] * 60 + default[1]


def is_quiet_hours(timestamp: float, config: Mapping[str, Any]) -> bool:
    if not bool(config.get("quiet_hours_enabled", True)):
        return False
    current = datetime.fromtimestamp(timestamp).hour * 60 + datetime.fromtimestamp(timestamp).minute
    start = _parse_hhmm(config.get("quiet_start", "00:00"), (0, 0))
    end = _parse_hhmm(config.get("quiet_end", "07:00"), (7, 0))
    if start == end:
        return False
    if start < end:
        return start <= current < end
    return current >= start or current < end


def proactive_count_24h(state: CompanionState, timestamp: float) -> int:
    cutoff = timestamp - 24 * 60 * 60
    return sum(
        1
        for item in state.recent_proactive_outcomes
        if item.get("status") == "sent"
        and _safe_float(item.get("sent_at"), 0.0) >= cutoff
    )


def can_send(
    state: CompanionState,
    timestamp: float,
    config: Mapping[str, Any],
) -> tuple[bool, str]:
    if not state.enabled:
        return False, "disabled"
    if not state.bound_umo:
        return False, "not_bound"
    if state.quiet_until > timestamp:
        return False, "quiet_until"
    if is_quiet_hours(timestamp, config):
        return False, "quiet_hours"

    min_gap = max(0, _safe_int(config.get("min_gap_minutes"), 45)) * 60
    if state.last_proactive_message_at and timestamp - state.last_proactive_message_at < min_gap:
        return False, "min_gap"

    daily_limit = max(0, _safe_int(config.get("max_proactive_per_24h"), 2))
    if daily_limit and proactive_count_24h(state, timestamp) >= daily_limit:
        return False, "daily_limit"
    return True, "ok"


def evaluate_opportunity(
    state: CompanionState,
    timestamp: float,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a candidate only; this function never sends or sleeps."""
    if state.pending_opportunity or not state.last_user_message_at:
        return None
    allowed, _ = can_send(state, timestamp, config)
    if not allowed:
        return None

    reasons: list[tuple[str, float]] = []
    # A single concrete reason should be strong enough to create an
    # opportunity.  The previous weights (0.38/0.22/0.18) meant that a
    # perfectly valid unfinished topic or emotional residue could never
    # reach the default 0.55 threshold on its own, so normal conversations
    # silently produced no pending task.
    if state.unfinished_topics:
        reasons.append(("unfinished_topic", 0.65))
    if state.memory_cues:
        reasons.append(("memory_cue", 0.58))
    if state.mood.label not in {"", "平静", "普通"}:
        reasons.append(("emotional_residue", 0.58))
    if state.current_scene:
        reasons.append(("scene_change", 0.18))

    score = sum(weight for _, weight in reasons)
    threshold = clamp(_safe_float(config.get("opportunity_threshold"), 0.55), 0.0, 1.0)
    natural_threshold = clamp(
        _safe_float(config.get("natural_greeting_threshold"), 0.72), 0.0, 1.0
    )

    if score >= threshold:
        if len(reasons) == 1:
            reason_type = reasons[0][0]
        else:
            reason_type = "mixed"
        return {
            "reason_type": reason_type,
            "score": round(min(1.0, score), 3),
            "created_at": timestamp,
        }

    # A natural greeting is deliberately harder to qualify and is only armed
    # after a real interaction summary exists. It is still a one-shot task,
    # never a periodic poll.
    if bool(config.get("allow_natural_greeting", True)) and state.last_interaction_summary:
        natural_score = natural_threshold
        if state.relationship_tone in {"亲近", "熟悉", "亲密"}:
            natural_score += 0.05
        if natural_score >= natural_threshold:
            return {
                "reason_type": "natural_greeting",
                "score": round(min(1.0, natural_score), 3),
                "created_at": timestamp,
            }
    return None


def choose_delay_seconds(
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    rng: random.Random | None = None,
) -> float:
    rng = rng or random
    minimum = max(0.0, _safe_float(config.get("min_delay_minutes"), 20.0)) * 60
    maximum = max(minimum, _safe_float(config.get("max_delay_minutes"), 240.0) * 60)
    reason_type = str(candidate.get("reason_type", "natural_greeting"))
    score = clamp(_safe_float(candidate.get("score"), 0.72), 0.0, 1.0)

    if reason_type == "natural_greeting":
        natural_minimum = max(
            0.0,
            _safe_float(config.get("natural_greeting_min_delay_minutes"), 60.0),
        ) * 60
        minimum = max(minimum, natural_minimum)
    elif reason_type in {"emotional_residue", "unfinished_topic", "mixed"}:
        # Stronger reasons are allowed to appear sooner, but remain delayed.
        ratio = 1.0 - clamp((score - 0.55) / 0.45, 0.0, 1.0)
        minimum = minimum + (maximum - minimum) * ratio * 0.35

    if maximum < minimum:
        maximum = minimum
    return rng.uniform(minimum, maximum)


def parse_decision(
    raw: Any,
    max_message_chars: int = 500,
) -> dict[str, Any]:
    payload = parse_json_object(raw) or {}
    send = payload.get("send") is True
    reason_type = str(payload.get("reason_type", "natural_greeting"))
    if reason_type not in REASON_TYPES:
        reason_type = "natural_greeting"
    message = normalize_message(payload.get("message", ""), max_message_chars)
    if not message:
        send = False
    return {
        "send": send,
        "reason_type": reason_type,
        "reason_summary": truncate(payload.get("reason_summary", ""), 240),
        "message": message,
        "confidence": clamp(_safe_float(payload.get("confidence"), 0.0), 0.0, 1.0),
        "next_cooldown_minutes": max(
            0, _safe_int(payload.get("next_cooldown_minutes"), 0)
        ),
    }


def make_outcome(
    status: str,
    timestamp: float,
    reason_type: str = "",
    reason_summary: str = "",
    message: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "status": status,
        "sent_at": timestamp if status == "sent" else 0.0,
        "created_at": timestamp,
        "reason_type": truncate(reason_type, 60),
        "reason_summary": truncate(reason_summary, 240),
        "message": truncate(message, 500),
        "error": truncate(error, 240),
    }


__all__ = [
    "CompanionState",
    "DEFAULT_CONFIG",
    "MoodState",
    "PLUGIN_ID",
    "StateStore",
    "can_send",
    "choose_delay_seconds",
    "evaluate_opportunity",
    "make_outcome",
    "merge_extraction",
    "normalize_message",
    "parse_decision",
    "parse_json_object",
    "proactive_count_24h",
    "truncate",
]
