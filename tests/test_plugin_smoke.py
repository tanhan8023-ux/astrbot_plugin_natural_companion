from __future__ import annotations

import asyncio
import sys
import types
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock


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


def onebot_event(umo: str, raw: dict[str, Any], text: str = "", components=None):
    """Match the adapter shape: notices can also be private AstrMessageEvents."""
    event = FakeEvent(umo, text)
    event.message_obj = types.SimpleNamespace(
        raw_message=raw,
        message=components or [],
        self_id=str(raw.get("self_id", "bot")),
        sender=types.SimpleNamespace(user_id=str(raw.get("user_id", "123"))),
        group_id=raw.get("group_id", ""),
    )
    return event


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

    async def test_typing_notices_keep_pending_task_and_state_unchanged(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        self.plugin.config["debug_logging"] = True
        await self.plugin._handle_command(FakeEvent(umo), "测试")
        pending_task = self.plugin._pending_task
        before = self.plugin.state.to_dict()
        persisted = self.state_path.read_bytes()
        issue = self.plugin._last_runtime_issue
        for raw in (
            {"post_type": "notice", "notice_type": "notify", "sub_type": "input_status", "event_type": 1},
            {"post_type": "notice", "notice_type": "notify", "sub_type": "input_status", "event_type": 2},
            {"notice_type": "notify", "sub_type": "input_status", "event_type": 2},
        ):
            with self.subTest(raw=raw):
                event = onebot_event(umo, raw)
                self.assertEqual([item async for item in self.plugin.on_private_message(event)], [])
                self.assertEqual(self.plugin.state.to_dict(), before)
                self.assertIs(self.plugin._pending_task, pending_task)
                self.assertEqual(self.plugin._message_revision, 0)
                self.assertIsNone(self.plugin._extract_task)
                self.assertEqual(self.plugin._last_runtime_issue, issue)
                self.assertFalse(event.stopped)  # Other plugins may consume the notice.
                self.assertEqual(self.state_path.read_bytes(), persisted)
        self.assertIn("输入状态通知", self.plugin._format_status_locked())

    async def test_non_message_events_cannot_bind_or_cancel(self):
        umo = "aiocqhttp:private:123"
        for raw in (
            {"post_type": "notice", "notice_type": "notify", "sub_type": "poke"},
            {"post_type": "notice", "notice_type": "friend_recall"},
            {"post_type": "meta_event", "meta_event_type": "heartbeat"},
            {"post_type": "request", "request_type": "friend"},
            {"post_type": "message_sent", "message_type": "private"},
            {"post_type": "message", "message_type": "private", "self_id": 99, "user_id": 99},
            {"post_type": "message", "message_type": "group", "group_id": 1},
        ):
            with self.subTest(raw=raw):
                event = onebot_event(umo, raw, "/主动聊天 开启")
                self.assertEqual([item async for item in self.plugin.on_private_message(event)], [])
                self.assertFalse(self.plugin.state.enabled)
                self.assertFalse(self.plugin.state.bound_umo)
                self.assertFalse(event.stopped)
        await self._bind(umo)
        self.plugin.config["test_delay_seconds"] = 60
        await self.plugin._handle_command(FakeEvent(umo), "测试")
        before = self.plugin.state.to_dict()
        # The observer must be safe even when invoked directly.
        await self.plugin._observe_bound_message(onebot_event(umo, {"post_type": "notice"}), "")
        self.assertEqual(self.plugin.state.to_dict(), before)

    async def test_empty_event_is_not_a_non_text_user_message(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        await self.plugin._handle_command(FakeEvent(umo), "测试")
        before = self.plugin.state.to_dict()
        for event in (FakeEvent(umo, ""), onebot_event(umo, {})):
            self.assertEqual([item async for item in self.plugin.on_private_message(event)], [])
            self.assertEqual(self.plugin.state.to_dict(), before)

    async def test_real_media_still_cancels_pending_opportunity(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        for kind in ("image", "record", "file", "face", "mface"):
            with self.subTest(kind=kind):
                await self.plugin._handle_command(FakeEvent(umo), "测试")
                pending_task = self.plugin._pending_task
                event = onebot_event(umo, {
                    "post_type": "message", "message_type": "private",
                    "self_id": 99, "user_id": 123,
                    "message": [{"type": kind, "data": {}}],
                })
                self.assertEqual([item async for item in self.plugin.on_private_message(event)], [])
                await asyncio.sleep(0)
                self.assertIsNone(self.plugin.state.pending_opportunity)
                self.assertTrue(pending_task.cancelled())
                self.assertGreater(self.plugin.state.last_user_message_at, 0)
        self.assertEqual(self.context.sent, [])

    async def test_typing_then_status_does_not_prevent_test_send(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 0
        self.context.conversation_manager = types.SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation"),
            get_conversation=AsyncMock(return_value=types.SimpleNamespace(
                history="[]", persona_id=None,
            )),
        )
        self.context.llm_generate = AsyncMock(return_value=types.SimpleNamespace(
            completion_text='{"send": true, "message": "之前那本书你读到哪一章了？", "confidence": 0.9}',
        ))
        event = FakeEvent(umo, "/主动聊天 测试")
        reply = [item async for item in self.plugin.on_private_message(event)]
        self.assertIn("测试主动机会", reply[0])
        task = self.plugin._pending_task
        notice = onebot_event(umo, {
            "post_type": "notice", "notice_type": "notify",
            "sub_type": "input_status", "event_type": 2,
        })
        self.assertEqual([item async for item in self.plugin.on_private_message(notice)], [])
        for command in ("/主动聊天 状态", "系尔主动聊天 状态"):
            event = FakeEvent(umo, command)
            self.assertTrue([item async for item in self.plugin.on_private_message(event)])
            self.assertIs(self.plugin._pending_task, task)
        await asyncio.wait_for(task, 1)
        self.context.llm_generate.assert_awaited_once()
        self.assertEqual(
            self.context.llm_generate.call_args.kwargs["system_prompt"],
            main.TEST_DECISION_SYSTEM_PROMPT,
        )
        self.assertEqual(len(self.context.sent), 1)
        self.assertEqual(self.plugin.state.recent_proactive_outcomes[-1]["status"], "sent")

    async def test_typing_during_model_decision_does_not_invalidate_it(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 0
        started = asyncio.Event()
        finish = asyncio.Event()

        async def slow_decision(*_args):
            started.set()
            await finish.wait()
            return await self._true_decision()

        self.plugin._decide_with_model = slow_decision
        await self.plugin._handle_command(FakeEvent(umo), "测试")
        task = self.plugin._pending_task
        await asyncio.wait_for(started.wait(), 1)
        self.assertIn("正在读取会话", self.plugin._format_status_locked())
        event = onebot_event(umo, {
            "post_type": "notice", "notice_type": "notify", "sub_type": "input_status",
        })
        self.assertEqual([item async for item in self.plugin.on_private_message(event)], [])
        self.assertEqual(self.plugin._message_revision, 0)
        finish.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(len(self.context.sent), 1)

    async def test_real_text_during_model_decision_still_cancels_it(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 0
        self.plugin.config["state_extract_debounce_seconds"] = 60
        started = asyncio.Event()

        async def slow_decision(*_args):
            started.set()
            await asyncio.Event().wait()
            return await self._true_decision()

        self.plugin._decide_with_model = slow_decision
        await self.plugin._handle_command(FakeEvent(umo), "测试")
        task = self.plugin._pending_task
        await asyncio.wait_for(started.wait(), 1)
        event = FakeEvent(umo, "我想接着刚才的话题聊")
        self.assertEqual([item async for item in self.plugin.on_private_message(event)], [])
        await asyncio.sleep(0)
        self.assertTrue(task.cancelled())
        self.assertEqual(self.context.sent, [])

    async def test_media_with_only_normalized_components_cancels(self):
        umo = await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        await self.plugin._handle_command(FakeEvent(umo), "测试")
        event = onebot_event(umo, {}, components=[types.SimpleNamespace(type="Image")])
        self.assertEqual([item async for item in self.plugin.on_private_message(event)], [])
        self.assertIsNone(self.plugin.state.pending_opportunity)

    async def test_status_reports_installed_version(self):
        umo = await self._bind()
        reply = await self.plugin._handle_command(FakeEvent(umo), "状态")
        self.assertIn(f"插件版本：{main.PLUGIN_VERSION}", reply)
        metadata = (Path(main.__file__).parent / "metadata.yaml").read_text(encoding="utf-8")
        self.assertIn(f"version: {main.PLUGIN_VERSION}", metadata)

    async def test_exact_pause_phrase_is_not_parsed_as_nickname_command(self):
        umo = await self._bind()
        for text in ("别主动找我", "不要主动找我", "先别主动找我"):
            with self.subTest(text=text):
                self.plugin.state.enabled = True
                reply = [item async for item in self.plugin.on_private_message(FakeEvent(umo, text))]
                self.assertIn("先不主动", reply[0])
                self.assertFalse(self.plugin.state.enabled)

    async def test_test_command_cancels_prior_state_extraction(self):
        umo = await self._bind()
        self.plugin.config["state_extract_debounce_seconds"] = 60
        await self.plugin._observe_bound_message(FakeEvent(umo), "普通聊天")
        extraction = self.plugin._extract_task
        await self.plugin._handle_command(FakeEvent(umo), "测试")
        await asyncio.sleep(0)
        self.assertTrue(extraction.cancelled())
        self.assertIsNone(self.plugin._extract_task)
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

