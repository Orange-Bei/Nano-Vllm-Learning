"""DPLLMEngine 的请求分发单测。

不启动真实 worker 进程：用 torch.mp.Pipe 造 fake worker 线程，
验证主进程的 round-robin + global seq_id + 消息序列化。
"""
import threading

import torch.multiprocessing as mp

from nanovllm.sampling_params import SamplingParams


class FakeWorker:
    """在独立线程里跑，接收主进程消息、echo ack 或 fake 响应。"""
    def __init__(self, child_conn):
        self.child_conn = child_conn
        self.received = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            try:
                msg = self.child_conn.recv()
            except EOFError:
                return
            self.received.append(msg)
            method = msg[0]
            if method == "exit":
                self.child_conn.send(None)
                return
            elif method == "add":
                self.child_conn.send(None)
            elif method == "is_finished":
                self.child_conn.send(True)
            elif method == "metrics_snapshot":
                self.child_conn.send({"step_samples": [], "completed_requests": {}, "total_preemptions": 0})
            elif method == "reset_metrics":
                self.child_conn.send(None)
            else:
                self.child_conn.send(None)


def _make_engine_with_fakes(n: int):
    """构造 DPLLMEngine，但 pipe/ps 用 fake worker 替换掉。"""
    from nanovllm.engine.dp_engine import DPLLMEngine
    engine = DPLLMEngine.__new__(DPLLMEngine)
    engine.data_parallel_size = n
    engine.next_dispatch_rank = 0
    engine.next_global_id = 0
    engine._exited = False
    engine.ps = []
    engine.pipes = []
    engine.tokenizer = None
    engine._fake_workers = []
    for _ in range(n):
        parent_conn, child_conn = mp.Pipe(duplex=True)
        engine.pipes.append(parent_conn)
        engine._fake_workers.append(FakeWorker(child_conn))
    return engine


class TestDispatch:
    def test_round_robin_3_workers(self):
        engine = _make_engine_with_fakes(3)
        sp = SamplingParams()
        gids = [engine.add_request([1, 2, 3], sp) for _ in range(6)]
        assert gids == [0, 1, 2, 3, 4, 5]
        assert len(engine._fake_workers[0].received) == 2
        assert len(engine._fake_workers[1].received) == 2
        assert len(engine._fake_workers[2].received) == 2
        for w in engine._fake_workers:
            for msg in w.received:
                assert msg[0] == "add"

    def test_message_format(self):
        engine = _make_engine_with_fakes(2)
        sp = SamplingParams(temperature=0.5, max_tokens=16)
        engine.add_request([10, 20, 30], sp)
        msg = engine._fake_workers[0].received[0]
        assert msg[0] == "add"
        assert msg[1] == 0
        assert msg[2] == [10, 20, 30]
        assert msg[3].temperature == 0.5
        assert msg[3].max_tokens == 16

    def test_is_finished_all_true(self):
        engine = _make_engine_with_fakes(3)
        assert engine.is_finished() is True

    def test_reset_metrics_broadcasts(self):
        engine = _make_engine_with_fakes(3)
        engine.reset_metrics()
        for w in engine._fake_workers:
            assert any(msg[0] == "reset_metrics" for msg in w.received)

    def test_exit_broadcasts(self):
        engine = _make_engine_with_fakes(2)
        engine.exit()
        for w in engine._fake_workers:
            assert any(msg[0] == "exit" for msg in w.received)
        engine.exit()  # 幂等
