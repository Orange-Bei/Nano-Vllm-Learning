"""端到端冒烟测试：需要 GPU 和 Qwen3-0.6B 模型。

如果模型路径不存在则自动跳过。llm fixture 在 conftest.py 中定义（session 级共享）。
"""
import os
import pytest

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/data/models/Qwen3-0.6B")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH),
    reason=f"Test model not found at {MODEL_PATH}",
)


def test_per_request_metrics_monotonicity(llm):
    from nanovllm import SamplingParams
    out = llm.generate(
        ["hello world", "introduce yourself"],
        SamplingParams(temperature=0.6, max_tokens=16),
    )
    assert len(out) == 2
    for o in out:
        m = o["metrics"]
        assert m is not None
        # 时间单调性
        assert m.arrival_time <= m.first_scheduled_time
        assert m.first_scheduled_time <= m.first_token_time
        assert m.first_token_time <= m.finish_time
        # token_times 长度 = completion token 数
        assert len(m.token_times) == m.num_completion_tokens
        # TTFT / E2E 都是正数
        assert m.ttft > 0
        assert m.e2e_latency > 0


def test_aggregate_metrics_nonempty(llm):
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(
        ["hello"] * 4,
        SamplingParams(temperature=0.6, max_tokens=16, ignore_eos=True),
    )
    agg = llm.get_aggregate_metrics()
    assert agg.total_requests == 4
    assert agg.ttft.avg > 0
    assert agg.decode_throughput > 0
    assert agg.prefill_throughput > 0
    # 至少有 prefill 和 decode 两类 step 样本
    assert any(s.is_prefill for s in agg.step_samples)
    assert any(not s.is_prefill for s in agg.step_samples)
    # prefill step 的 num_batched_tokens 必须 > 0（否则吞吐算不对）
    prefill_samples = [s for s in agg.step_samples if s.is_prefill]
    assert all(s.num_batched_tokens > 0 for s in prefill_samples)


def test_to_dict_is_json_serializable(llm):
    import json
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(["hello"], SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True))
    agg = llm.get_aggregate_metrics()
    # 不抛异常即通过
    json.dumps(agg.to_dict())


def test_summary_table_contains_expected_rows(llm):
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(["hello"], SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True))
    table = llm.get_aggregate_metrics().summary_table()
    for key in ["TTFT", "TPOT", "E2E", "Queue", "Prefill", "Decode",
                "Inter-token latency"]:
        assert key in table, f"summary_table 缺少 '{key}' 行"


def test_reset_metrics_clears_state(llm):
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(["hello"], SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True))
    assert llm.get_aggregate_metrics().total_requests == 1
    llm.reset_metrics()
    assert llm.get_aggregate_metrics().total_requests == 0
