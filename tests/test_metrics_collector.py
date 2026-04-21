import pytest

from nanovllm.engine.metrics import (
    EngineMetrics,
    MetricsCollector,
    Percentiles,
    RequestMetrics,
    StepSample,
)


class FakeBlockManager:
    def __init__(self, free: int, used: int):
        self.free_block_ids = list(range(free))
        self.used_block_ids = set(range(used))


class FakeSeq:
    """只实现 MetricsCollector 需要读取的属性。"""
    def __init__(self, seq_id, arrival, first_sched, first_tok, finish,
                 token_times, n_prompt, n_completion, preempt):
        self.seq_id = seq_id
        self.arrival_time = arrival
        self.first_scheduled_time = first_sched
        self.first_token_time = first_tok
        self.finish_time = finish
        self.token_times = token_times
        self.num_prompt_tokens = n_prompt
        self._n_completion = n_completion
        self.preemption_count = preempt

    @property
    def num_completion_tokens(self):
        return self._n_completion

    def as_request_metrics(self):
        return RequestMetrics(
            arrival_time=self.arrival_time,
            first_scheduled_time=self.first_scheduled_time,
            first_token_time=self.first_token_time,
            finish_time=self.finish_time,
            token_times=list(self.token_times),
            num_prompt_tokens=self.num_prompt_tokens,
            num_completion_tokens=self._n_completion,
            preemption_count=self.preemption_count,
        )


def make_seq(seq_id=1, arrival=0.0, n_completion=5):
    return FakeSeq(
        seq_id=seq_id,
        arrival=arrival,
        first_sched=arrival + 0.01,
        first_tok=arrival + 0.1,
        finish=arrival + 1.0,
        token_times=[arrival + 0.1 * i for i in range(1, n_completion + 1)],
        n_prompt=10,
        n_completion=n_completion,
        preempt=0,
    )


class TestMetricsCollector:
    def test_record_finished_stores_request_metrics(self):
        c = MetricsCollector()
        s = make_seq(seq_id=1)
        c.record_finished(s)
        assert c.get_request_metrics(1) is not None
        assert c.get_request_metrics(1).num_completion_tokens == 5

    def test_record_preemption_increments_total(self):
        c = MetricsCollector()
        s = make_seq(seq_id=1)
        c.record_preemption(s, 0.5)
        c.record_preemption(s, 0.8)
        assert c.total_preemptions == 2

    def test_record_step_appends_sample(self):
        c = MetricsCollector()
        bm = FakeBlockManager(free=100, used=20)
        s = make_seq()
        c.record_step(0.0, 0.005, [s], is_prefill=True, block_manager=bm, num_batched_tokens=256)
        assert len(c.step_samples) == 1
        sample = c.step_samples[0]
        assert sample.is_prefill is True
        assert sample.num_seqs == 1
        assert sample.num_batched_tokens == 256
        assert sample.num_free_blocks == 100
        assert sample.num_used_blocks == 20
        assert sample.step_duration == pytest.approx(0.005)


class TestEngineMetricsBuild:
    def test_empty_collector(self):
        c = MetricsCollector()
        em = c.build()
        assert em.total_requests == 0
        assert em.total_preemptions == 0
        import math
        assert math.isnan(em.ttft.avg)

    def test_single_request(self):
        c = MetricsCollector()
        s = make_seq(seq_id=1, n_completion=5)
        c.record_finished(s)
        em = c.build()
        assert em.total_requests == 1
        assert em.ttft.avg == pytest.approx(0.1)

    def test_throughput_from_steps(self):
        c = MetricsCollector()
        bm = FakeBlockManager(free=100, used=0)
        # 一个 prefill step: 1024 tokens, 0.05s
        c.record_step(0.0, 0.05, [make_seq()], is_prefill=True, block_manager=bm,
                      num_batched_tokens=1024)
        # 一个 decode step: 4 seqs, 0.01s -> 4 tok / 0.01s = 400 tok/s
        c.record_step(0.06, 0.07, [make_seq(), make_seq(seq_id=2),
                                   make_seq(seq_id=3), make_seq(seq_id=4)],
                      is_prefill=False, block_manager=bm, num_batched_tokens=4)
        # finish 一个请求以确保 total_requests > 0
        c.record_finished(make_seq(seq_id=1))
        em = c.build()
        assert em.prefill_throughput == pytest.approx(1024 / 0.05, rel=0.01)
        assert em.decode_throughput == pytest.approx(4 / 0.01, rel=0.01)

    def test_to_dict_is_json_serializable(self):
        import json
        c = MetricsCollector()
        c.record_finished(make_seq(seq_id=1))
        em = c.build()
        d = em.to_dict()
        # 能被 json.dumps 序列化（不抛异常）
        json.dumps(d)

    def test_summary_table_returns_string(self):
        c = MetricsCollector()
        c.record_finished(make_seq(seq_id=1))
        em = c.build()
        table = em.summary_table()
        assert isinstance(table, str)
        assert "TTFT" in table
        assert "TPOT" in table
