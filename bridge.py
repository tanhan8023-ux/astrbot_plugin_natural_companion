"""Read-only view of Alive Persona's SheShe conversation bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhoneEvent:
    id: str
    role: str
    content: str
    timestamp: str


@dataclass(frozen=True)
class PhoneSnapshot:
    binding_id: str
    events: tuple[PhoneEvent, ...]


def bound_qq_id(umo: str) -> str:
    """Extract the QQ ID from an aiocqhttp private event's persisted UMO.

    AstrBot uses the configured platform instance ID (not necessarily the
    adapter name) and FriendMessage for private chat sessions. Keep accepting
    older private-format bindings while refusing group/session variants.
    """
    parts = umo.split(":")
    return (
        parts[2]
        if len(parts) == 3
        and parts[0]
        and parts[1] in ("FriendMessage", "private")
        and parts[2].isdigit()
        else ""
    )


def read_phone_snapshot(path: Path, qq_id: str) -> PhoneSnapshot:
    """Reject mismatched identities and malformed files without modifying the bridge."""
    with path.open("r", encoding="utf-8") as stream:
        data: Any = json.load(stream)
    if not isinstance(data, dict) or data.get("schemaVersion") != 1:
        raise ValueError("unsupported bridge schema")
    binding = data.get("binding")
    if not isinstance(binding, dict) or str(binding.get("userId", "")) != qq_id:
        raise ValueError("bridge QQ binding does not match this private chat")
    binding_id = str(binding.get("bindingId") or "").strip()
    if not binding_id:
        raise ValueError("bridge has no binding ID")
    raw_events = data.get("recentEvents")
    if not isinstance(raw_events, list):
        raise ValueError("bridge recent events are unavailable")
    events = []
    for raw in raw_events[-80:]:
        if not isinstance(raw, dict) or raw.get("source") != "sheshe":
            continue
        event_id = str(raw.get("id") or "").strip()[:160]
        role = raw.get("role")
        content = raw.get("content")
        if not event_id or role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        events.append(PhoneEvent(event_id, role, content.strip()[:1000], str(raw.get("timestamp") or "")[:60]))
    return PhoneSnapshot(binding_id, tuple(events))


def phone_context(snapshot: PhoneSnapshot) -> str:
    """Ephemeral, bounded cross-device context for model prompts only."""
    return "\n".join(
        f"{item.timestamp} {'user' if item.role == 'user' else 'assistant'}: {item.content[:500]}"
        for item in snapshot.events[-12:]
    )[:6000]
