import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Sampler requires CUDA",
)


def _make_sampler():
    from nanovllm.layers.sampler import Sampler
    return Sampler().cuda()


def _tensor_1d(vals, dtype):
    return torch.tensor(vals, dtype=dtype, device="cuda")


class TestSamplerGreedy:
    def test_all_greedy_equals_argmax(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(4, 100, device="cuda")
        expected = logits.argmax(dim=-1)

        out = sampler(
            logits.clone(),
            _tensor_1d([0.0, 0.0, 0.0, 0.0], torch.float32),
            _tensor_1d([-1, -1, -1, -1], torch.int64),
            _tensor_1d([1.0, 1.0, 1.0, 1.0], torch.float32),
        )
        assert torch.equal(out, expected)

    def test_top_k_one_equals_argmax(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(4, 100, device="cuda")
        expected = logits.argmax(dim=-1)

        # top_k=1 且 temperature>0：Gumbel 路径里只有 max 存活，采样结果恒等 argmax
        out = sampler(
            logits.clone(),
            _tensor_1d([0.8, 0.8, 0.8, 0.8], torch.float32),
            _tensor_1d([1, 1, 1, 1], torch.int64),
            _tensor_1d([1.0, 1.0, 1.0, 1.0], torch.float32),
        )
        assert torch.equal(out, expected)


class TestSamplerTopK:
    def test_top_k_five_restricts_choices(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(1, 100, device="cuda")
        top5 = set(logits.topk(5, dim=-1).indices[0].tolist())

        sampled = set()
        for i in range(200):
            out = sampler(
                logits.clone(),
                _tensor_1d([1.0], torch.float32),
                _tensor_1d([5], torch.int64),
                _tensor_1d([1.0], torch.float32),
            )
            sampled.add(out.item())

        assert sampled.issubset(top5), f"sampled={sampled}, top5={top5}"


class TestSamplerTopP:
    def test_top_p_tiny_keeps_only_max(self):
        sampler = _make_sampler()
        # 构造一个单一 token 概率极大的 logits
        logits = torch.full((1, 50), -10.0, device="cuda")
        logits[0, 7] = 10.0  # token 7 占绝大多数概率

        for i in range(100):
            out = sampler(
                logits.clone(),
                _tensor_1d([1.0], torch.float32),
                _tensor_1d([-1], torch.int64),
                _tensor_1d([0.5], torch.float32),
            )
            assert out.item() == 7


class TestSamplerDefaults:
    def test_all_disabled_matches_legacy_gumbel(self):
        """全禁用 top_k/top_p/rp，结果应服从 softmax(logits/temp) 分布。"""
        sampler = _make_sampler()
        torch.manual_seed(42)
        logits = torch.randn(1, 20, device="cuda")
        temp = 0.8

        expected_probs = torch.softmax(logits[0] / temp, dim=-1).cpu().numpy()

        N = 5000
        counts = [0] * 20
        for _ in range(N):
            out = sampler(
                logits.clone(),
                _tensor_1d([temp], torch.float32),
                _tensor_1d([-1], torch.int64),
                _tensor_1d([1.0], torch.float32),
            )
            counts[out.item()] += 1

        # 简化的 χ² 检验：每个 bin 的经验频率与期望概率误差 < 0.05
        for i in range(20):
            empirical = counts[i] / N
            expected = float(expected_probs[i])
            assert abs(empirical - expected) < 0.05, (
                f"token {i}: empirical={empirical:.3f} vs expected={expected:.3f}"
            )


class TestSamplerMixedBatch:
    def test_mixed_greedy_and_sampled(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(2, 100, device="cuda")
        row0_argmax = logits[0].argmax().item()

        # 行 0: greedy (temp=0)
        # 行 1: 常规采样 (temp=1.0)
        for _ in range(100):
            out = sampler(
                logits.clone(),
                _tensor_1d([0.0, 1.0], torch.float32),
                _tensor_1d([-1, -1], torch.int64),
                _tensor_1d([1.0, 1.0], torch.float32),
            )
            assert out[0].item() == row0_argmax  # greedy 永远 argmax

    def test_mixed_top_k_values(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(2, 100, device="cuda")
        top1_row0 = logits[0].argmax().item()
        top5_row1 = set(logits[1].topk(5).indices.tolist())

        row1_sampled = set()
        for _ in range(200):
            out = sampler(
                logits.clone(),
                _tensor_1d([0.8, 0.8], torch.float32),
                _tensor_1d([1, 5], torch.int64),  # 行 0 top_k=1，行 1 top_k=5
                _tensor_1d([1.0, 1.0], torch.float32),
            )
            assert out[0].item() == top1_row0
            row1_sampled.add(out[1].item())

        assert row1_sampled.issubset(top5_row1)
