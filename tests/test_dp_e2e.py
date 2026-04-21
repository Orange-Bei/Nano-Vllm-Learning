"""DP 端到端活性测试。需要 ≥2 GPU。

独立 module fixture：DP=2 的 LLM 实例；teardown 显式 exit() 释放所有 worker。
"""
import os

import pytest
import torch

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/data/models/Qwen3-0.6B")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH) or torch.cuda.device_count() < 2,
    reason=f"需要模型 {MODEL_PATH} 和 ≥2 GPU",
)


@pytest.fixture(scope="module")
def dp_llm():
    from nanovllm import LLM
    instance = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=1024,
        tensor_parallel_size=1,
        data_parallel_size=2,
    )
    yield instance
    instance.exit()


def test_dp_generate_completes(dp_llm):
    """基础活性：16 条混合长度 prompt 都能完成。"""
    from nanovllm import SamplingParams

    prompts = [f"hello world {i}" for i in range(8)] + \
              [f"introduce yourself briefly (case {i})" for i in range(8)]
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    outputs = dp_llm.generate(prompts, sp)

    assert len(outputs) == 16
    for o in outputs:
        assert o["text"], f"空输出: {o}"
        assert len(o["token_ids"]) > 0
        assert o["metrics"] is not None
        assert o["metrics"].ttft > 0


def test_dp_aggregate_metrics_has_multi_rank_steps(dp_llm):
    """聚合 metrics 应该能看到来自 2 个 worker 的 step。"""
    from nanovllm import SamplingParams

    dp_llm.reset_metrics()
    prompts = [f"hi {i}" for i in range(8)]
    sp = SamplingParams(temperature=0.0, max_tokens=4)
    dp_llm.generate(prompts, sp)

    agg = dp_llm.get_aggregate_metrics()
    assert len(agg.step_samples) > 0
    assert agg.total_requests == 8


def test_dp_reset_metrics(dp_llm):
    """reset_metrics 后 aggregate metrics 清零。"""
    from nanovllm import SamplingParams

    dp_llm.generate(["warm"], SamplingParams(temperature=0.0, max_tokens=2))
    agg_before = dp_llm.get_aggregate_metrics()
    assert agg_before.total_requests >= 1

    dp_llm.reset_metrics()
    agg_after = dp_llm.get_aggregate_metrics()
    assert agg_after.total_requests == 0
    assert agg_after.step_samples == []
