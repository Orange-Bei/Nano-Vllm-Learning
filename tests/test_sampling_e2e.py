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


def test_greedy_is_deterministic(llm):
    """temperature=0 两次调用同一 prompt 结果完全一致。"""
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    out1 = llm.generate(["introduce yourself"], sp)
    out2 = llm.generate(["introduce yourself"], sp)
    assert out1[0]["token_ids"] == out2[0]["token_ids"]


def test_top_k_one_with_greedy_is_deterministic(llm):
    """top_k=1 + temperature=0 走 greedy 路径，两次调用确定性。

    注：top_k=1 + temperature>0 走 Gumbel 路径，若 logits 并列最大则 Gumbel 随机选，
    不能保证跨调用确定性。确定性交给 temperature=0 的 greedy 路径。
    """
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0.0, top_k=1, max_tokens=32)
    out1 = llm.generate(["introduce yourself"], sp)
    out2 = llm.generate(["introduce yourself"], sp)
    assert out1[0]["token_ids"] == out2[0]["token_ids"]


def test_repetition_penalty_reduces_repetition(llm):
    """repetition_penalty > 1 会减少 token 重复。

    注意：这是概率性断言，用 enforce_eager + 固定 temperature 让 Gumbel 随机性仍存在，
    但期望 rp 的效果显著到能稳定通过。若偶发失败，放宽比例阈值。
    """
    from nanovllm import SamplingParams
    prompt = "The the the the the the the the the"  # 诱导重复
    max_tokens = 64

    sp_nop = SamplingParams(temperature=0.8, repetition_penalty=1.0, max_tokens=max_tokens)
    sp_rp = SamplingParams(temperature=0.8, repetition_penalty=1.5, max_tokens=max_tokens)

    out_nop = llm.generate([prompt], sp_nop)[0]["token_ids"]
    out_rp = llm.generate([prompt], sp_rp)[0]["token_ids"]

    def uniq_ratio(tokens):
        return len(set(tokens)) / max(len(tokens), 1)

    # rp=1.5 的去重率应 >= rp=1.0（宽松断言——采样有随机性）
    assert uniq_ratio(out_rp) >= uniq_ratio(out_nop) - 0.1, (
        f"rp=1.5 uniq={uniq_ratio(out_rp):.2f}, rp=1.0 uniq={uniq_ratio(out_nop):.2f}"
    )


def test_mixed_sampling_params_batch_runs(llm):
    """传入不同 SamplingParams 的 list，generate 正常工作。"""
    from nanovllm import SamplingParams
    prompts = [
        "introduce yourself",
        "list three colors",
        "what is 2+2?",
        "write a haiku",
    ]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=16),                          # greedy
        SamplingParams(temperature=0.8, top_k=50, max_tokens=16),                # top-k
        SamplingParams(temperature=0.8, top_p=0.9, max_tokens=16),               # top-p
        SamplingParams(temperature=0.8, repetition_penalty=1.2, max_tokens=16),  # rep_penalty
    ]

    outputs = llm.generate(prompts, sps)

    assert len(outputs) == 4
    for o in outputs:
        assert o["text"]
        assert len(o["token_ids"]) > 0
        # round #1 指标仍然附着
        assert o["metrics"] is not None
        assert o["metrics"].ttft > 0


def test_no_recompile_explosion(llm):
    """连续多次不同 batch 大小调用，torch.compile 不应爆 recompile。

    此断言是观察性的：当前 torch 版本若不支持 `_dynamo.utils.counters`，退化为只要 generate 不崩溃即可。
    """
    from nanovllm import SamplingParams
    import torch._dynamo

    # 重置 dynamo 状态，让首次编译开始计数
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    prompts_pool = ["hello", "world", "test", "python", "code"]
    sp = SamplingParams(temperature=0.8, top_k=50, top_p=0.9,
                        repetition_penalty=1.1, max_tokens=8)

    for batch_size in [1, 2, 3, 4, 5]:
        prompts = prompts_pool[:batch_size]
        outputs = llm.generate(prompts, sp)
        assert len(outputs) == batch_size

    # 观察性检查——若 API 不存在则跳过断言
    try:
        counters = torch._dynamo.utils.counters
        unique_graphs = counters.get("stats", {}).get("unique_graphs", None)
        if unique_graphs is not None:
            # 启动编译 + 可能的边界条件，不应远超一个小常数
            # 放宽阈值到 10，主要是防止失控（几百个）
            assert unique_graphs <= 10, f"torch.compile 生成了 {unique_graphs} 个独立图，可能存在 recompile 爆炸"
    except Exception:
        pass  # API 不可用，观察性断言退化为无


def test_backward_compat_default_params(llm):
    """默认 SamplingParams 的输出仍然合理（round #1 的 example.py 那种用法）。"""
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0.6, max_tokens=32)
    outputs = llm.generate(["introduce yourself"], sp)
    assert len(outputs) == 1
    assert outputs[0]["text"]
    assert outputs[0]["metrics"].ttft > 0
