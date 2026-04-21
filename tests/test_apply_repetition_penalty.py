import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="apply_repetition_penalty requires CUDA",
)


class _FakeSeq:
    """仅用于测试，复用 Sequence 的 duck-typed 接口。"""
    def __init__(self, token_ids, repetition_penalty):
        self.token_ids = token_ids
        self.repetition_penalty = repetition_penalty


def _make_logits(batch_size, vocab_size, device="cuda"):
    torch.manual_seed(0)
    return torch.randn(batch_size, vocab_size, device=device)


class TestApplyRepetitionPenalty:
    def test_penalty_one_is_noop(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = _make_logits(2, 100)
        before = logits.clone()
        seqs = [_FakeSeq([1, 2, 3], 1.0), _FakeSeq([4, 5, 6], 1.0)]
        apply_repetition_penalty(logits, seqs)
        assert torch.allclose(before, logits)

    def test_positive_logit_divided(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.zeros(1, 10, device="cuda")
        logits[0, 3] = 4.0  # 正 logit
        seqs = [_FakeSeq([3], 2.0)]  # rp=2
        apply_repetition_penalty(logits, seqs)
        assert logits[0, 3].item() == pytest.approx(2.0)  # 4 / 2

    def test_negative_logit_multiplied(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.zeros(1, 10, device="cuda")
        logits[0, 3] = -4.0  # 负 logit
        seqs = [_FakeSeq([3], 2.0)]  # rp=2
        apply_repetition_penalty(logits, seqs)
        assert logits[0, 3].item() == pytest.approx(-8.0)  # -4 * 2

    def test_repeated_token_penalized_once(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.zeros(1, 10, device="cuda")
        logits[0, 3] = 4.0
        seqs = [_FakeSeq([3, 3, 3, 3, 3], 2.0)]  # token 3 出现 5 次
        apply_repetition_penalty(logits, seqs)
        # scatter 幂等，不叠加惩罚
        assert logits[0, 3].item() == pytest.approx(2.0)  # 4 / 2，不是 4 / 32

    def test_mixed_batch_only_nondefault_row_modified(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.randn(3, 10, device="cuda")
        before = logits.clone()
        seqs = [
            _FakeSeq([1, 2], 1.0),   # rp=1.0 禁用，跳过
            _FakeSeq([3, 4], 2.0),   # rp=2.0 生效
            _FakeSeq([5, 6], 1.0),   # rp=1.0 禁用，跳过
        ]
        apply_repetition_penalty(logits, seqs)
        assert torch.allclose(before[0], logits[0])   # 行 0 不变
        assert not torch.allclose(before[1], logits[1])  # 行 1 变了
        assert torch.allclose(before[2], logits[2])   # 行 2 不变

    def test_untouched_tokens_unchanged(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.randn(1, 100, device="cuda")
        before = logits.clone()
        seqs = [_FakeSeq([5, 10, 15], 2.0)]
        apply_repetition_penalty(logits, seqs)
        # 只有索引 5/10/15 的 logit 应该改变
        for i in range(100):
            if i in (5, 10, 15):
                assert not torch.equal(logits[0, i], before[0, i])
            else:
                assert torch.equal(logits[0, i], before[0, i])
