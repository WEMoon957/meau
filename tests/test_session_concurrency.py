"""SessionManager 每会话锁测试 - 防止并发历史覆写（lost-update 防护）。

缺陷背景：
  SessionManager.chat 执行 load -> agent.chat (5-8s LLM) -> save 三步，
  若同一 session_id 的两个请求并发执行（多 tab / 双击 / 前端重试），
  会各自读到旧历史并相互覆写，导致一整轮对话历史丢失。
  修复后通过每会话锁串行化，第二个并发请求立即获得 SessionBusyError
  （API 层映射 429），杜绝历史覆写。
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

from langchain_core.messages import AIMessage, HumanMessage
from session_manager import SessionBusyError, SessionManager


class _FakeAgent:
    """模拟无状态 OrderingAgent：可控延迟，记录调用次数。"""

    MAX_HISTORY = 20

    def __init__(self, delay=0.0):
        self.delay = delay
        self.call_count = 0
        self._count_lock = threading.Lock()

    def chat(self, user_input, history):
        with self._count_lock:
            self.call_count += 1
        if self.delay:
            time.sleep(self.delay)
        resp = f"reply:{user_input}"
        return resp, [HumanMessage(content=user_input), AIMessage(content=resp)]


def test_concurrent_same_session_one_rejected():
    """同一 session_id 的两个并发请求：恰好一个成功，一个被 SessionBusyError 拒绝。"""
    agent = _FakeAgent(delay=0.3)  # 模拟 LLM 延迟，确保时间窗重叠
    manager = SessionManager(agent, redis_client=None)

    results = {"ok": [], "busy": 0}
    barrier = threading.Barrier(2)

    def worker(msg):
        barrier.wait()
        try:
            results["ok"].append(manager.chat("s1", msg))
        except SessionBusyError:
            results["busy"] += 1

    t1 = threading.Thread(target=worker, args=("m1",))
    t2 = threading.Thread(target=worker, args=("m2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results["ok"]) == 1, f"expected 1 ok, got {results['ok']}"
    assert results["busy"] == 1, f"expected 1 busy, got {results['busy']}"


def test_concurrent_no_history_loss():
    """并发后历史完整：成功的那条请求的对话已正确写入，未因覆写丢失。"""
    agent = _FakeAgent(delay=0.2)
    manager = SessionManager(agent, redis_client=None)

    barrier = threading.Barrier(2)
    outcomes = []

    def worker(msg):
        barrier.wait()
        try:
            manager.chat("s2", msg)
            outcomes.append(msg)
        except SessionBusyError:
            outcomes.append("busy")

    t1 = threading.Thread(target=worker, args=("hello",))
    t2 = threading.Thread(target=worker, args=("world",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    history = manager._load_history("s2")
    # 成功的请求应写入恰好 2 条消息（1 Human + 1 AI），未被另一个请求覆写
    assert len(history) == 2, f"expected 2 messages, got {len(history)}"


def test_sequential_calls_accumulate_history():
    """串行调用正常累积历史，锁不影响正常流程。"""
    agent = _FakeAgent()
    manager = SessionManager(agent, redis_client=None)

    manager.chat("s3", "hello")
    manager.chat("s3", "world")

    history = manager._load_history("s3")
    assert len(history) == 4, f"expected 4 messages, got {len(history)}"


def test_different_sessions_concurrent_ok():
    """不同 session_id 的并发请求互不阻塞。"""
    agent = _FakeAgent(delay=0.2)
    manager = SessionManager(agent, redis_client=None)

    errors = []
    barrier = threading.Barrier(2)

    def worker(sid, msg):
        barrier.wait()
        try:
            manager.chat(sid, msg)
        except SessionBusyError:
            errors.append(sid)

    t1 = threading.Thread(target=worker, args=("a", "m1"))
    t2 = threading.Thread(target=worker, args=("b", "m2"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"different sessions should not block, got: {errors}"


def test_lock_released_after_success():
    """成功请求后锁已释放，后续同 session 请求可正常获取。"""
    agent = _FakeAgent()
    manager = SessionManager(agent, redis_client=None)

    manager.chat("s4", "first")
    manager.chat("s4", "second")  # 若锁未释放，此处会 SessionBusyError

    history = manager._load_history("s4")
    assert len(history) == 4


def test_lock_released_after_exception():
    """agent.chat 抛异常时锁也必须释放（finally），不锁死会话。"""
    class _BoomAgent:
        MAX_HISTORY = 20
        def chat(self, user_input, history):
            raise RuntimeError("boom")

    manager = SessionManager(_BoomAgent(), redis_client=None)

    try:
        manager.chat("s5", "x")
    except RuntimeError:
        pass

    # 锁应已释放，可正常获取
    agent = _FakeAgent()
    manager.agent = agent
    manager.chat("s5", "recovered")
    history = manager._load_history("s5")
    assert len(history) == 2


if __name__ == "__main__":
    test_concurrent_same_session_one_rejected()
    print("PASS test_concurrent_same_session_one_rejected")
    test_concurrent_no_history_loss()
    print("PASS test_concurrent_no_history_loss")
    test_sequential_calls_accumulate_history()
    print("PASS test_sequential_calls_accumulate_history")
    test_different_sessions_concurrent_ok()
    print("PASS test_different_sessions_concurrent_ok")
    test_lock_released_after_success()
    print("PASS test_lock_released_after_success")
    test_lock_released_after_exception()
    print("PASS test_lock_released_after_exception")
    print("\nAll tests passed.")
