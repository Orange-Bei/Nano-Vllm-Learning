"""Chunked prefill 推广的端到端活性测试。

独立的 module fixture：`chunked_llm` 用小 max_num_batched_tokens 强制 chunk。
teardown 显式 .exit()，释放 dist 供后续模块使用。

注意：把所有断言合并到一次 generate 调用里——多次 generate 会在
@torch.compile 的 rotary_embedding.forward 上触发 dynamo 动态形状限制
（s1 != s0 报错），不是 scheduler 改动的 bug 但会误伤测试。
"""
import os
import pytest

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/data/models/Qwen3-0.6B")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH),
    reason=f"Test model not found at {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def chunked_llm():
    from nanovllm import LLM
    instance = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,   # 小预算，强制任何较长 prompt 被 chunk
        tensor_parallel_size=1,
    )
    yield instance
    instance.exit()
    # 确保 GPU 显存被回收，否则后续模块的 LLM 算 num_kvcache_blocks 会拿不到预算
    import gc
    import torch
    del instance
    gc.collect()
    torch.cuda.empty_cache()


def test_chunked_prefill_activity(chunked_llm):
    """一次 generate 覆盖：长 prompt 触发 chunk + budget 打满 + mixed batch 完成。"""
    from nanovllm import SamplingParams

    chunked_llm.reset_metrics()

    prompts = [
        "hello " * 300,                  # 超 256 预算，触发 chunking
        "hi",                             # 短，正常
        "introduce yourself briefly",    # 中短，正常
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    outputs = chunked_llm.generate(prompts, sp)

    # 所有 seq 完成且输出非空
    assert len(outputs) == 3
    for o in outputs:
        assert o["text"], f"输出为空：{o}"
        assert len(o["token_ids"]) > 0
        assert o["metrics"] is not None
        assert o["metrics"].ttft > 0

    # 长 prompt 确实 tokenize 超预算
    assert outputs[0]["metrics"].num_prompt_tokens >= 256, (
        f"prompt 只 tokenize 到 {outputs[0]['metrics'].num_prompt_tokens} token，"
        "e2e 没真正触发 chunking；调大 prompt 重复次数"
    )

    # 至少一个 prefill step 打满 budget（证明 chunk 生效）
    agg = chunked_llm.get_aggregate_metrics()
    prefill_samples = [s for s in agg.step_samples if s.is_prefill]
    assert prefill_samples, "至少应该有一个 prefill step"
    max_batched = max(s.num_batched_tokens for s in prefill_samples)
    assert max_batched == 256, (
        f"prefill step 的 num_batched_tokens 最大值 = {max_batched}, 预期 256"
    )
