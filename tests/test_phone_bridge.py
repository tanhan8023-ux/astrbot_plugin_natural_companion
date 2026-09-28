from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from test_plugin_smoke import FakeContext, FakeEvent

import main
from bridge import bound_qq_id, read_phone_snapshot
from core import DEFAULT_CONFIG


class PhoneBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path("tests") / "_runtime"
        self.root.mkdir(parents=True, exist_ok=True)
        unique = uuid.uuid4().hex
        self.state_path = self.root / f"phone_state_{unique}.json"
        self.bridge_path = self.root / f"sheshe_bridge_{unique}.json"
        self.original_state_path = main._plugin_state_path
        main._plugin_state_path = lambda: self.state_path
        self.context = FakeContext()
        self.config = dict(DEFAULT_CONFIG)
        self.config.update({
            "phone_bridge_enabled": True,
            "phone_bridge_path": str(self.bridge_path),
            "state_extract_debounce_seconds": 0,
            "min_delay_minutes": 1,
            "max_delay_minutes": 1,
            "quiet_hours_enabled": False,
            "min_gap_minutes": 0,
        })
        self.plugin = main.NaturalCompanionPlugin(self.context, self.config)
        self.plugin._extract_state_with_model = self._extract

    async def asyncTearDown(self):
        await self.plugin.terminate()
        main._plugin_state_path = self.original_state_path
        for path in (self.state_path, self.state_path.with_name(self.state_path.name + ".tmp"), self.bridge_path):
            path.unlink(missing_ok=True)

    async def _extract(self, _umo, message, _state=None):
        return {
            "mood": {"label": "好奇"},
            "current_scene": "小手机里提到了未完的话题",
            "unfinished_topics": ["想继续聊的事"],
            "memory_cues": [],
        }

    def _event(self, event_id: str, role: str = "user", content: str = "我们明天继续聊") -> dict:
        return {
            "id": event_id, "role": role, "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(), "source": "sheshe",
        }

    def _write(self, events: list[dict], *, qq: str = "123", binding: str = "phone-1"):
        self.bridge_path.write_text(json.dumps({
            "schemaVersion": 1,
            "binding": {"userId": qq, "bindingId": binding},
            "recentEvents": events,
        }, ensure_ascii=False), encoding="utf-8")

    async def _bind(self, events=None):
        self._write(events or [])
        await self.plugin._handle_command(FakeEvent("aiocqhttp:private:123"), "开启")

    async def _finish_extract(self):
        task = self.plugin._extract_task
        if task is not None:
            await asyncio.wait_for(task, 2)

    async def test_parser_rejects_wrong_qq_and_ignores_qq_echoes(self):
        self.assertEqual(bound_qq_id("aiocqhttp:private:123"), "123")
        self.assertEqual(bound_qq_id("my_qq_bot:FriendMessage:123"), "123")
        self.assertEqual(bound_qq_id("aiocqhttp:group:123"), "")
        self.assertEqual(bound_qq_id("my_qq_bot:GroupMessage:123"), "")
        self.assertEqual(bound_qq_id(":FriendMessage:123"), "")
        self.assertEqual(bound_qq_id("my_qq_bot:FriendMessage:abc"), "")
        self._write([self._event("phone"), {**self._event("qq"), "source": "astrbot"}])
        self.assertEqual([item.id for item in read_phone_snapshot(self.bridge_path, "123").events], ["phone"])
        with self.assertRaises(ValueError):
            read_phone_snapshot(self.bridge_path, "456")

    async def test_real_private_umo_connects_and_observes_phone_message(self):
        self._write([self._event("old")])
        result = await self.plugin._handle_command(FakeEvent("my_qq_bot:FriendMessage:123"), "开启")
        self.assertIn("已开启", result)
        await self._finish_extract()
        self.assertEqual(self.plugin._format_status_locked().split("小手机互通：")[1].splitlines()[0], "已连接")
        self._write([self._event("old"), self._event("new")])
        self.assertTrue(await self.plugin._sync_phone_bridge())
        await self._finish_extract()
        self.assertIsNotNone(self.plugin.state.pending_opportunity)

    async def test_initial_history_refreshes_status_without_sending(self):
        await self._bind([self._event("old")])
        await self._finish_extract()
        self.assertEqual(self.plugin.state.unfinished_topics, ["想继续聊的事"])
        self.assertGreater(self.plugin.state.last_phone_message_at, 0)
        self.assertIsNone(self.plugin.state.pending_opportunity)
        self.assertEqual(self.context.sent, [])
        self.assertNotIn("我们明天继续聊", self.state_path.read_text(encoding="utf-8"))

    async def test_fresh_phone_user_creates_one_opportunity_and_deduplicates(self):
        await self._bind([self._event("old")])
        await self._finish_extract()
        self._write([self._event("old"), self._event("fresh")])
        await self.plugin._sync_phone_bridge()
        await self._finish_extract()
        pending = self.plugin.state.pending_opportunity
        self.assertIsNotNone(pending)
        task = self.plugin._pending_task
        await self.plugin._sync_phone_bridge()
        self.assertIs(self.plugin._pending_task, task)
        self.assertEqual(self.plugin.state.pending_opportunity, pending)

    async def test_new_phone_user_cancels_pending_and_assistant_does_not(self):
        await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        await self.plugin._handle_command(FakeEvent("aiocqhttp:private:123"), "测试")
        old_task = self.plugin._pending_task
        self._write([self._event("assistant", "assistant", "我记着这件事")])
        await self.plugin._sync_phone_bridge()
        self.assertIs(self.plugin._pending_task, old_task)
        self._write([self._event("assistant", "assistant"), self._event("new")])
        await self.plugin._sync_phone_bridge()
        await asyncio.sleep(0)
        self.assertTrue(old_task.cancelled())
        self.assertIsNone(self.plugin.state.pending_opportunity)
        await self._finish_extract()
        self.assertIsNotNone(self.plugin.state.pending_opportunity)
        self.assertNotEqual(self.plugin._pending_task, old_task)

    async def test_phone_turn_during_model_decision_prevents_stale_send(self):
        await self._bind()
        self.plugin.config["test_delay_seconds"] = 0
        self.plugin.config["state_extract_debounce_seconds"] = 60
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_decision(*_args):
            started.set()
            await release.wait()
            return await self._true_decision()

        self.plugin._decide_with_model = slow_decision
        await self.plugin._handle_command(FakeEvent("aiocqhttp:private:123"), "测试")
        task = self.plugin._pending_task
        await asyncio.wait_for(started.wait(), 2)
        # The file changes while the model is busy, before the watcher runs.
        self._write([self._event("during")])
        release.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.context.sent, [])
        self.assertIsNone(self.plugin.state.pending_opportunity)

    async def _true_decision(self):
        return {"send": True, "message": "想起我们之前聊的事。", "reason_type": "unfinished_topic",
                "reason_summary": "旧话题", "next_cooldown_minutes": 0}

    async def test_phone_turn_during_extraction_supersedes_stale_result(self):
        await self._bind()
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_extract(*_args):
            started.set()
            await release.wait()
            return {"current_scene": "过时的话题", "unfinished_topics": ["旧事"]}

        self.plugin._extract_state_with_model = slow_extract
        self._write([self._event("first")])
        await self.plugin._sync_phone_bridge()
        await asyncio.wait_for(started.wait(), 2)
        self._write([self._event("first"), self._event("second")])
        await self.plugin._sync_phone_bridge()
        self.plugin._extract_state_with_model = self._extract
        release.set()
        await self._finish_extract()
        self.assertEqual(self.plugin.state.current_scene, "小手机里提到了未完的话题")
        self.assertNotIn("旧事", self.plugin.state.unfinished_topics)

    async def test_missing_corrupt_and_wrong_binding_pause_opportunity(self):
        await self._bind()
        for failure in ("missing", "corrupt", "wrong"):
            with self.subTest(failure=failure):
                self._write([])
                await self.plugin._sync_phone_bridge()
                self.plugin.config["test_delay_seconds"] = 60
                await self.plugin._handle_command(FakeEvent("aiocqhttp:private:123"), "测试")
                task = self.plugin._pending_task
                if failure == "missing":
                    self.bridge_path.unlink()
                elif failure == "corrupt":
                    self.bridge_path.write_text("{broken", encoding="utf-8")
                else:
                    self._write([], qq="999")
                self.assertFalse(await self.plugin._sync_phone_bridge())
                await asyncio.sleep(0)
                self.assertTrue(task.cancelled())
                self.assertIsNone(self.plugin.state.pending_opportunity)
                self.assertIn("暂停主动机会", self.plugin._format_status_locked())

    async def test_restart_and_clear_do_not_replay_old_phone_messages(self):
        await self._bind([self._event("old")])
        await self._finish_extract()
        await self.plugin.terminate()
        self.plugin = main.NaturalCompanionPlugin(self.context, self.config)
        self.plugin._extract_state_with_model = self._extract
        await self.plugin.initialize()
        self.assertIsNone(self.plugin.state.pending_opportunity)
        self.assertEqual(self.context.sent, [])
        await self.plugin._handle_command(FakeEvent("aiocqhttp:private:123"), "清除记忆")
        self.assertIsNone(self.plugin.state.pending_opportunity)
        self.assertEqual(self.plugin.state.phone_event_ids, ["old"])
        await self.plugin._sync_phone_bridge()
        self.assertIsNone(self.plugin.state.pending_opportunity)

    async def test_restart_restores_waiting_opportunity_without_new_phone_turn(self):
        await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        await self.plugin._handle_command(FakeEvent("aiocqhttp:private:123"), "测试")
        previous = dict(self.plugin.state.pending_opportunity)
        await self.plugin.terminate()

        self.plugin = main.NaturalCompanionPlugin(self.context, self.config)
        self.plugin._extract_state_with_model = self._extract
        await self.plugin.initialize()

        self.assertEqual(self.plugin.state.pending_opportunity, previous)
        self.assertIsNotNone(self.plugin._pending_task)
        self.assertFalse(self.plugin._pending_task.done())
        self.assertEqual(self.context.sent, [])

    async def test_restart_phone_turn_cancels_old_opportunity_before_restore(self):
        await self._bind()
        self.plugin.config["test_delay_seconds"] = 60
        await self.plugin._handle_command(FakeEvent("aiocqhttp:private:123"), "测试")
        old_id = self.plugin.state.pending_opportunity["id"]
        await self.plugin.terminate()
        self._write([self._event("offline")])

        self.plugin = main.NaturalCompanionPlugin(self.context, self.config)
        self.plugin._extract_state_with_model = self._extract
        await self.plugin.initialize()
        self.assertIsNone(self.plugin.state.pending_opportunity)
        await self._finish_extract()

        self.assertIsNotNone(self.plugin.state.pending_opportunity)
        self.assertNotEqual(self.plugin.state.pending_opportunity["id"], old_id)
        self.assertEqual(self.context.sent, [])

    async def test_phone_pause_and_rebind_reset_the_cursor(self):
        await self._bind()
        self._write([self._event("pause", content="别主动找我")])
        await self.plugin._sync_phone_bridge()
        self.assertFalse(self.plugin.state.enabled)
        self.assertIsNone(self.plugin.state.pending_opportunity)
        await self.plugin._handle_command(FakeEvent("aiocqhttp:private:456", admin=True), "重绑")
        self.assertEqual(self.plugin.state.bound_umo, "aiocqhttp:private:456")
        self.assertFalse(self.plugin.state.phone_cursor_ready)
        self.assertIn("不可用", self.plugin._format_status_locked())


if __name__ == "__main__":
    unittest.main()
