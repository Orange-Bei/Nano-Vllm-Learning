import torch
from torch import nn


def apply_repetition_penalty(logits: torch.Tensor, seqs: list) -> None:
    """In-place 对 logits 应用 repetition_penalty。

    历史范围 = seq.token_ids（prompt + completion），对齐 vLLM / HF 行为。
    rp == 1.0 的 seq 整行跳过（最常见情况零 GPU op）。
    Scatter 幂等：同一 token 在历史中重复出现不会叠加惩罚。
    """
    for i, seq in enumerate(seqs):
        rp = seq.repetition_penalty
        if rp == 1.0:
            continue
        tokens = torch.tensor(seq.token_ids, device=logits.device, dtype=torch.int64)
        selected = logits[i].gather(0, tokens)
        penalized = torch.where(selected < 0, selected * rp, selected / rp)
        logits[i].scatter_(0, tokens, penalized)


class Sampler(nn.Module):
    # greedy 判定阈值；temperatures.clamp(min=EPS) 防 Gumbel 路径除零。
    EPS: float = 1e-5

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        V = logits.size(-1)

        # --- greedy 路径：原始 logits 上 argmax（不受 top_k/top_p 影响，对齐 vLLM 语义） ---
        greedy_tokens = logits.argmax(dim=-1)

        # --- top_k：一次 sort 拿到第 k 大的阈值 ---
        k = torch.where(top_k <= 0, V, top_k).clamp(max=V)          # -1 -> V（no-op）
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        thresholds = sorted_logits.gather(-1, (k - 1).unsqueeze(-1))
        masked = logits.masked_fill(logits < thresholds, float('-inf'))

        # --- top_p：复用 sorted_logits 做 nucleus 裁剪 ---
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        remove_sorted = (cumsum - sorted_probs) > top_p.unsqueeze(-1)  # top_p=1.0 -> 全 False
        remove = torch.zeros_like(remove_sorted).scatter_(-1, sorted_idx, remove_sorted)
        masked = masked.masked_fill(remove, float('-inf'))

        # --- temperature + Gumbel-max ---
        scaled = masked.div(temperatures.clamp(min=self.EPS).unsqueeze(-1))
        probs = scaled.softmax(dim=-1)
        sampled_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        # --- 逐行选：greedy seq 取 argmax，其他取 Gumbel ---
        return torch.where(temperatures < self.EPS, greedy_tokens, sampled_tokens)
