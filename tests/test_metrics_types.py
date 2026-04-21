import math
import pytest

from nanovllm.engine.metrics import Percentiles, RequestMetrics, StepSample


class TestPercentiles:
    def test_empty_samples_returns_nan(self):
        p = Percentiles.from_samples([])
        assert math.isnan(p.avg)
        assert math.isnan(p.min)
        assert math.isnan(p.p50)
        assert math.isnan(p.p90)
        assert math.isnan(p.p95)
        assert math.isnan(p.p99)
        assert math.isnan(p.max)

    def test_single_sample(self):
        p = Percentiles.from_samples([1.5])
        assert p.avg == 1.5
        assert p.min == 1.5
        assert p.p50 == 1.5
        assert p.p99 == 1.5
        assert p.max == 1.5

    def test_multiple_samples(self):
        p = Percentiles.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
        assert p.avg == 3.0
        assert p.min == 1.0
        assert p.max == 5.0
        assert p.p50 == 3.0


class TestRequestMetrics:
    def _make(self, arrival=0.0, first_sched=0.01, first_tok=0.1, finish=1.0,
              token_times=None, n_prompt=10, n_completion=5, preempt=0):
        if token_times is None:
            token_times = [0.1, 0.3, 0.5, 0.7, 1.0]
        return RequestMetrics(
            arrival_time=arrival,
            first_scheduled_time=first_sched,
            first_token_time=first_tok,
            finish_time=finish,
            token_times=token_times,
            num_prompt_tokens=n_prompt,
            num_completion_tokens=n_completion,
            preemption_count=preempt,
        )

    def test_ttft(self):
        m = self._make(arrival=0.0, first_tok=0.1)
        assert m.ttft == pytest.approx(0.1)

    def test_tpot_normal(self):
        m = self._make(first_tok=0.1, finish=1.0, n_completion=5)
        # (1.0 - 0.1) / (5 - 1) = 0.225
        assert m.tpot == pytest.approx(0.225)

    def test_tpot_single_completion_returns_zero(self):
        m = self._make(n_completion=1, finish=0.1)
        assert m.tpot == 0.0

    def test_tpot_zero_completion_returns_zero(self):
        m = self._make(n_completion=0, finish=0.1)
        assert m.tpot == 0.0

    def test_e2e_latency(self):
        m = self._make(arrival=0.0, finish=1.0)
        assert m.e2e_latency == pytest.approx(1.0)

    def test_queue_time(self):
        m = self._make(arrival=0.0, first_sched=0.05)
        assert m.queue_time == pytest.approx(0.05)

    def test_prefill_time(self):
        m = self._make(first_sched=0.05, first_tok=0.12)
        assert m.prefill_time == pytest.approx(0.07)

    def test_decode_time(self):
        m = self._make(first_tok=0.1, finish=1.0)
        assert m.decode_time == pytest.approx(0.9)

    def test_inter_token_intervals(self):
        m = self._make(token_times=[0.1, 0.3, 0.5])
        assert m.inter_token_intervals == pytest.approx([0.2, 0.2])

    def test_inter_token_intervals_single_token(self):
        m = self._make(token_times=[0.1])
        assert m.inter_token_intervals == []


class TestStepSample:
    def test_fields(self):
        s = StepSample(
            timestamp=0.0,
            is_prefill=True,
            num_seqs=4,
            num_batched_tokens=1024,
            num_free_blocks=100,
            num_used_blocks=20,
            step_duration=0.005,
        )
        assert s.num_seqs == 4
        assert s.is_prefill is True
