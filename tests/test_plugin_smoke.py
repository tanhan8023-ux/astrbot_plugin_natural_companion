from __future__ import annotations

import asyncio
import sys
import types
import unittest
import uuid
from pathlib import Path
from typing import Any


def _install_astrbot_stubs() -> None:
    """Install only the tiny AstrBot surface needed for local plugin smoke tests."""
    try:
        import astrbot.api.event  # noqa: F401
        import astrbot.api.star  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")

    class Logger:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    class AstrMessageEvent:
        pass

    class MessageChain:
        def message(self, _value):
            return self

    def decorator(*_args, **_kwargs):
        def wrap(fn):
            return fn

        return wrap

    class PlatformAdapterType:
        AIOCQHTTP = "aiocqhttp"

    class EventMessageType:
        PRIVATE_MESSAGE = "private"

    platform_adapter_type = PlatformAdapterType
    event_message_type = EventMessageType

    class Filter:
        pass

    Filter.PlatformAdapterType = PlatformAdapterType
    Filter.EventMessageType = EventMessageType
    Filter.platform_adapter_type = staticmethod(decorator)
    Filter.event_message_type = staticmethod(decorator)

    class Context:
        pass

    class Star:
        def __init__(self, context):
            self.context = context

    api.logger = Logger()
    event.AstrMessageEvent = AstrMessageEvent
    event.MessageChain = MessageChain
    event.filter = Filter()
    star.Context = Context
    star.Star = Star
    astrbot.__path__ = []

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.star": star,
        }
    )


_install_astrbot_stubs()

import main  # noqa: E402
from core import CompanionState, DEFAULT_CONFIG  # noqa: E402


class FakeEvent:
    def __init__(self, umo: str, text: str = "你好", admin: bool = False):
        self.unified_msg_origin = umo
        self.message_str = text
        self.admin = admin
        self.stopped = False

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text: str):
        return text

    def get_message_outline(self):
        return self.message_str

    async def is_admin(self):
        return self.admin


class FakeContext:
    def __init__(self):
        self.sent: list[tuple[str, Any]] = []
        self.fail_send = False

    async def get_current_chat_provider_id(self, **_kwargs):
        return "provider"

    async def llm_generate(self, **_kwargs):
        return types.SimpleNamespace(completion_text='{"send": false}')

    async def send_message(self, umo, message):
        if self.fail_send:
            raise RuntimeError("simulated OneBot failure")
        self.sent.append((umo, message))
        return True


class PluginSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime_dir = Path("tests") / "_runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / f"plugin_{uuid.uuid4().hex}.json"
        self.context = FakeContext()
        original_path = main._plugin_state_path
        main._plugin_state_path = lambda: self.state_path
        self.addAsyncCleanup(self._cleanup_plugin)
        self.addCleanup(lambda: setattr(main, "_plugin_state_path", original_path))
        config = dict(DEFAULT_CONFIG)
        config.update(
            {
                "min_delay_minutes": 0,
                "max_delay_minutes": 0,
                "min_gap_minutes": 0,
                "max_proactive_per_24h": 2,
                "quiet_hours_enabled": False,
                "state_extract_debounce_seconds": 0,
            }
        )
        self.plugin = main.NaturalCompanionPlugin(self.context, config)

    async def _cleanup_plugin(self):
        await self.plugin.terminate()
        for path in (self.state_path, self.state_path.with_name(self.state_path.name + ".tmp")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    async def _bind(self, umo: str = "aiocqhttp:private:123"):
        result = await self.plugin._handle_command(FakeEvent(umo), "开启")
        self.assertIn("已开启", result)
        return umo

    async def test_enable_and_non_bound_message_is_ignored(self):
        umo = await self._bind()
        await self.plugin._observe_bound_message(FakeEvent("aiocqhttp:private:other"), "别人的消息")
        self.assertEqual(self.plugin.state.bound_umo, umo)
        self.assertEqual(self.plugin.state.last_user_message_at, 0.0)
        self.assertIsNone(self.plugin._extract_task)

    async def test_explicit_test_command_arms_short_one_shot_opportunity(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        result = await self.plugin._handle_command(FakeEvent(umo), "测试")
        self.assertIn("测试主动机会", result)
        self.assertIsNotNone(self.plugin.state.pending_opportunity)
        self.assertTrue(self.plugin.state.pending_opportunity["test_mode"])

    async def test_rebind_clears_previous_private_chat_memory(self):
        old_umo = await self._bind("aiocqhttp:private:old")
        self.plugin.state.current_scene = "旧会话情景"
        self.plugin.state.unfinished_topics = ["旧话题"]
        self.plugin.state.memory_cues = ["旧记忆"]
        self.plugin.state.last_interaction_summary = "旧摘要"
        result = await self.plugin._handle_command(
            FakeEvent("aiocqhttp:private:new", admin=True), "重绑"
        )
        self.assertIn("重新绑定", result)
        self.assertEqual(self.plugin.state.bound_umo, "aiocqhttp:private:new")
        self.assertTrue(self.plugin.state.enabled)
        self.assertEqual(self.plugin.state.current_scene, "")
        self.assertEqual(self.plugin.state.unfinished_topics, [])
        self.assertEqual(self.plugin.state.memory_cues, [])
        self.assertEqual(self.plugin.state.last_interaction_summary, "")
        self.assertNotEqual(old_umo, self.plugin.state.bound_umo)

    async def test_new_message_cancels_waiting_task(self):
        umo = await self._bind()
        async with self.plugin._state_lock:
            self.plugin.state.last_user_message_at = 1.0
            self.plugin.state.last_interaction_summary = "有一个未完话题"
            self.plugin.state.pending_opportunity = {
                "id": "waiting",
                "created_at": 1.0,
                "due_at": 9999999999.0,
                "reason_type": "unfinished_topic",
                "score": 0.8,
                "status": "waiting",
            }
            self.plugin._start_pending_task("waiting", 9999999999.0)
        old_task = self.plugin._pending_task
        await self.plugin._observe_bound_message(FakeEvent(umo), "我又发了一条")
        await asyncio.sleep(0)
        self.assertIsNone(self.plugin.state.pending_opportunity)
        self.assertTrue(old_task.cancelled())
        self.assertEqual(self.context.sent, [])

    async def test_missing_conversation_safely_skips_model_send(self):
        await self._bind()
        decision = await self.plugin._decide_with_model(
            "aiocqhttp:private:123", self.plugin.state, {"reason_type": "unfinished_topic"}
        )
        self.assertFalse(decision["send"])
        self.assertEqual(self.context.sent, [])

    async def test_model_send_false_does_not_send(self):
        umo = await self._bind()
        now = __import__("time").time()
        self.plugin.state.last_user_message_at = now - 10
        self.plugin.state.pending_opportunity = {
            "id": "no-send",
            "created_at": now,
            "due_at": now - 1,
            "reason_type": "unfinished_topic",
            "score": 0.8,
            "status": "waiting",
        }
        self.plugin._decide_with_model = self._false_decision
        await self.plugin._wait_and_decide("no-send", now - 1)
        self.assertEqual(self.context.sent, [])
        self.assertIsNone(self.plugin.state.pending_opportunity)
        self.assertEqual(self.plugin.state.recent_proactive_outcomes[-1]["status"], "skipped")

    async def _false_decision(self, *_args, **_kwargs):
        return {
            "send": False,
            "reason_type": "natural_greeting",
            "reason_summary": "现在不够自然",
            "message": "",
            "confidence": 0.9,
            "next_cooldown_minutes": 0,
        }

    async def test_successful_send_is_recorded(self):
        umo = await self._bind()
        now = __import__("time").time()
        self.plugin.state.last_user_message_at = now - 10
        self.plugin.state.pending_opportunity = {
            "id": "send",
            "created_at": now,
            "due_at": now - 1,
            "reason_type": "unfinished_topic",
            "score": 0.8,
            "status": "waiting",
        }
        self.plugin._decide_with_model = self._true_decision
        await self.plugin._wait_and_decide("send", now - 1)
        self.assertEqual(len(self.context.sent), 1)
        self.assertIsNone(self.plugin.state.pending_opportunity)
        self.assertEqual(self.plugin.state.recent_proactive_outcomes[-1]["status"], "sent")

    async def _true_decision(self, *_args, **_kwargs):
        return {
            "send": True,
            "reason_type": "unfinished_topic",
            "reason_summary": "接上未完话题",
            "message": "刚才那个话题我又想起来了。",
            "confidence": 0.9,
            "next_cooldown_minutes": 0,
        }

    async def test_send_failure_is_recorded_without_crashing(self):
        await self._bind()
        self.context.fail_send = True
        now = __import__("time").time()
        self.plugin.state.last_user_message_at = now - 10
        self.plugin.state.pending_opportunity = {
            "id": "failed-send",
            "created_at": now,
            "due_at": now - 1,
            "reason_type": "unfinished_topic",
            "score": 0.8,
            "status": "waiting",
        }
        self.plugin._decide_with_model = self._true_decision
        await self.plugin._wait_and_decide("failed-send", now - 1)
        self.assertIsNone(self.plugin.state.pending_opportunity)
        self.assertEqual(self.plugin.state.recent_proactive_outcomes[-1]["status"], "failed")

    async def test_persisted_sending_opportunity_is_not_replayed(self):
        await self._bind()
        self.plugin.state.pending_opportunity = {
            "id": "in-flight",
            "created_at": 1.0,
            "due_at": 9999999999.0,
            "reason_type": "unfinished_topic",
            "score": 0.8,
            "status": "sending",
        }
        self.plugin.store.save(self.plugin.state)
        restored = main.NaturalCompanionPlugin(self.context, self.plugin.config)
        try:
            await restored.initialize()
            self.assertIsNone(restored.state.pending_opportunity)
            self.assertEqual(self.context.sent, [])
        finally:
            await restored.terminate()


if __name__ == "__main__":
    unittest.main()

