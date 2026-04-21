import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) # 这个mask用于标记输入的token ids中哪些在当前分区的词汇范围内；对于在当前分区范围内的token ids，mask对应位置为True，否则为False；这个mask将用于后续的计算，确保只有当前分区负责处理的token ids被映射到嵌入向量，而其他分区的token ids将被忽略。
            x = mask * (x - self.vocab_start_idx) # 将输入的token ids转换为当前分区的局部索引；对于在当前分区范围内的token ids，减去vocab_start_idx将它们映射到0到num_embeddings_per_partition-1的范围内；对于不在当前分区范围内的token ids，mask对应位置为False，乘以0后会被置为0，这样在后续的embedding计算中就不会对这些token ids产生影响。
        y = F.embedding(x, self.weight) # 根据输入的token ids和当前分区的嵌入权重计算嵌入向量；对于在当前分区范围内的token ids，embedding函数会根据局部索引从self.weight中查找对应的嵌入向量；对于不在当前分区范围内的token ids，由于它们被置为0，embedding函数会返回一个全零的嵌入向量；这样就实现了词汇表并行化，每个分区只负责处理自己范围内的token ids。
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # 将mask扩展到嵌入向量的维度，以便在后续的all_reduce操作中正确地聚合来自不同分区的嵌入向量；对于在当前分区范围内的token ids，mask对应位置为True，乘以嵌入向量后保持不变；对于不在当前分区范围内的token ids，mask对应位置为False，乘以嵌入向量后会被置为全零，这样在后续的all_reduce操作中就不会对这些token ids产生影响。
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
