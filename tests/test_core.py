from __future__ import annotations

import random
import unittest
import uuid
from datetime import datetime
from pathlib import Path

from core import (
    CompanionState,
    DEFAULT_CONFIG,
    MoodState,
    StateStore,
    can_send,
    choose_delay_seconds,
    evaluate_opportunity,
    merge_extraction,
    normalize_message,
    parse_decision,
    parse_json_object,
)


def config(**overrides):
    value = dict(DEFAULT_CONFIG)
    value.update(overrides)
    if "quiet_hours_enabled" not in overrides:
        value["quiet_hours_enabled"] = False
    return value


def enabled_state() -> CompanionState:
    return CompanionState(
        enabled=True,
        bound_umo="aiocqhttp:private:123",
        last_user_message_at=1_000.0,
        last_interaction_summary="刚刚聊到一本没看完的书",
    )


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.runtime_dir = Path("tests") / "_runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / f"state_{uuid.uuid4().hex}.json"

    def tearDown(self):
        for path in (self.state_path, self.state_path.with_name(self.state_path.name + ".tmp")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def test_json_parser_accepts_fence_and_prefix(self):
        self.assertEqual(
            parse_json_object('```json\n{"send": true}\n```'), {"send": True}
        )
        self.assertEqual(
            parse_json_object('结果如下： {"value": 1} 结束'), {"value": 1}
        )
        self.assertIsNone(parse_json_object("not json"))

    def test_state_round_trip_and_clear(self):
        store = StateStore(self.state_path)
        state = enabled_state()
        state.mood = MoodState("好奇", 0.7, 0.8, "想继续聊下去")
        state.unfinished_topics = ["一本书"]
        store.save(state)

        restored = store.load()
        self.assertTrue(restored.enabled)
        self.assertEqual(restored.bound_umo, state.bound_umo)
        self.assertEqual(restored.mood.label, "好奇")
        self.assertEqual(restored.unfinished_topics, ["一本书"])

        restored.clear_plugin_memory()
        self.assertTrue(restored.enabled)
        self.assertEqual(restored.bound_umo, state.bound_umo)
        self.assertEqual(restored.unfinished_topics, [])
        self.assertEqual(restored.last_user_message_at, 0.0)

    def test_corrupt_state_falls_back_to_default(self):
        self.state_path.write_text("{broken", encoding="utf-8")
        self.assertFalse(StateStore(self.state_path).load().enabled)

    def test_invalid_extraction_keeps_existing_state(self):
        state = enabled_state()
        state.current_scene = "原来的情景"
        state.unfinished_topics = ["原来的话题"]
        before = state.to_dict()
        merge_extraction(state, parse_json_object("这不是 JSON"))
        self.assertEqual(state.to_dict(), before)

    def test_merge_extraction_only_updates_valid_structured_fields(self):
        state = enabled_state()
        merge_extraction(
            state,
            {
                "mood": {"label": "轻松", "valence": 1.5, "arousal": -1},
                "current_scene": "晚饭后闲聊",
                "unfinished_topics": ["旅行计划", {"summary": "电影"}],
                "memory_cues": ["用户喜欢猫"],
                "relationship_tone": "亲近",
                "user_preferences": ["不喜欢连续追问"],
            },
        )
        self.assertEqual(state.mood.label, "轻松")
        self.assertEqual(state.mood.valence, 1.0)
        self.assertEqual(state.mood.arousal, 0.0)
        self.assertEqual(state.current_scene, "晚饭后闲聊")
        self.assertEqual(state.unfinished_topics, ["旅行计划", "电影"])
        self.assertEqual(state.relationship_tone, "亲近")

    def test_strong_reason_creates_one_shot_candidate(self):
        state = enabled_state()
        state.unfinished_topics = ["旅行计划"]
        state.memory_cues = ["周末要做的事"]
        candidate = evaluate_opportunity(state, 2_000.0, config())
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["reason_type"], "mixed")
        self.assertGreaterEqual(candidate["score"], 0.55)
        self.assertLessEqual(candidate["score"], 1.0)

    def test_natural_greeting_requires_interaction_summary(self):
        state = enabled_state()
        state.unfinished_topics = []
        state.memory_cues = []
        state.current_scene = ""
        state.last_interaction_summary = ""
        self.assertIsNone(evaluate_opportunity(state, 2_000.0, config()))

        state.last_interaction_summary = "最近聊过工作"
        candidate = evaluate_opportunity(state, 2_000.0, config())
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["reason_type"], "natural_greeting")

    def test_natural_greeting_has_longer_delay_than_minimum(self):
        candidate = {"reason_type": "natural_greeting", "score": 0.72}
        delay = choose_delay_seconds(candidate, config(), random.Random(1))
        self.assertGreaterEqual(delay, 60 * 60)

    def test_decision_parser_rejects_invalid_or_empty_message(self):
        decision = parse_decision('{"send": true, "message": ""}')
        self.assertFalse(decision["send"])

        decision = parse_decision(
            '{"send": true, "reason_type": "bad", "message": "你好！"}'
        )
        self.assertTrue(decision["send"])
        self.assertEqual(decision["reason_type"], "natural_greeting")

    def test_message_is_shortened_to_three_sentences(self):
        message = normalize_message("第一句。第二句！第三句？第四句。", 500)
        self.assertEqual(message, "第一句。第二句！第三句？")

    def test_send_limits_apply(self):
        state = enabled_state()
        state.last_proactive_message_at = 1_950.0
        allowed, reason = can_send(state, 2_000.0, config(min_gap_minutes=1))
        self.assertFalse(allowed)
        self.assertEqual(reason, "min_gap")

        state.last_proactive_message_at = 0.0
        state.recent_proactive_outcomes = [
            {"status": "sent", "sent_at": 1_500.0},
            {"status": "sent", "sent_at": 1_600.0},
        ]
        allowed, reason = can_send(
            state,
            2_000.0,
            config(max_proactive_per_24h=2),
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "daily_limit")

    def test_quiet_hours_apply(self):
        now = datetime.now().astimezone()
        quiet_local = now.replace(
            hour=1, minute=30, second=0, microsecond=0
        ).timestamp()
        state = enabled_state()
        allowed, reason = can_send(
            state,
            quiet_local,
            config(
                quiet_hours_enabled=True,
                quiet_start="00:00",
                quiet_end="07:00",
            ),
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "quiet_hours")

    def test_pending_opportunity_blocks_duplicate_candidates(self):
        state = enabled_state()
        state.unfinished_topics = ["一个话题"]
        state.pending_opportunity = {"id": "already-pending", "due_at": 9_999}
        self.assertIsNone(evaluate_opportunity(state, 2_000.0, config()))


if __name__ == "__main__":
    unittest.main()
