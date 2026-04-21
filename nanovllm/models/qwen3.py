import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear( # 同时计算查询、键、值的线性变换，减少计算量；输入维度是 hidden_size，输出维度是 q_size + 2 * kv_size，其中 q_size 是查询的维度，kv_size 是键和值的维度；输出的前 q_size 个维度对应查询，接下来的 kv_size 个维度对应键，最后的 kv_size 个维度对应值。
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear( # 输出投影层：将自注意力机制的输出映射回隐藏状态的维度；
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias: #Qwen3相比 LLaMA 多出来的一步，论文里叫QK-Norm，目的是稳住 attention 的 softmax 数值分布，让深层训练更稳。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module): # 前馈神经网络：包含一个门控线性单元和一个下投影层；

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear( # 门控线性单元：同时计算门控值和激活值，减少计算量；输出维度是 intermediate_size 的两倍，一半用于门控值，一半用于激活值
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear( # 下投影层：将激活值映射回隐藏状态的维度；输入维度是 intermediate_size，输出维度是 hidden_size
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul() # 激活函数：使用SiluAndMul，它是SiLU激活函数和元素级乘法的组合；它首先对输入的激活值应用SiLU函数，然后将结果与门控值进行元素级乘法，得到最终的输出；这种激活函数可以提高模型的非线性表达能力和性能。

    def forward(self, x): # 前馈神经网络的前向传播：首先通过门控线性单元计算门控值和激活值，然后对激活值应用SiluAndMul激活函数，最后通过下投影层将结果映射回隐藏状态的维度，得到前馈神经网络的输出。
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention( 
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP( 
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None: 
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module): # 主干：embed + N 层 decoder + 最终 norm

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size) # 嵌入层：将输入的token ids映射到对应的嵌入向量；它使用了VocabParallelEmbedding类，这个类实现了词汇表并行化，即将词汇表划分成多个部分，每个部分由不同的设备处理，从而加速嵌入层的计算；嵌入层的权重可以通过weight_loader方法加载预训练的权重。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])# 解码器层：由多个Qwen3DecoderLayer组成，每个解码器层包含一个自注意力机制和一个前馈神经网络；每个解码器层还包含两个RMSNorm层，分别用于自注意力机制之前和之后的残差连接；解码器层的输入是嵌入层的输出，输出是经过自注意力机制和前馈神经网络处理后的隐藏状态；解码器层之间通过残差连接传递信息。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 最终的RMSNorm层：在所有解码器层之后对隐藏状态进行归一化；它使用了RMSNorm类，这个类实现了均方根归一化，可以提高模型的训练稳定性和性能；最终的RMSNorm层的权重可以通过weight_loader方法加载预训练的权重。

    def forward( 
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers: # 依次通过每个解码器层进行处理；每个解码器层的输入是当前的隐藏状态和残差，输出是更新后的隐藏状态和新的残差；残差用于连接解码器层之间的信息流，帮助模型更好地捕捉长距离依赖关系；最后通过最终的RMSNorm层对隐藏状态进行归一化，得到模型的输出。
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module): # 最外层的模型类，包含了整个模型的结构，包括嵌入层、多个解码器层和语言建模头；它的forward方法定义了输入数据如何通过这些层进行前向传播；它还定义了一个compute_logits方法，用于从最后的隐藏状态计算输出的logits，这些logits可以用来预测下一个token的概率分布。
    packed_modules_mapping = { # 这个字典定义了模型中哪些模块的权重是共享的，以及它们在不同模块之间是如何共享的；例如，"q_proj"模块的权重在"qkv_proj"模块中的"q"部分被共享，"k_proj"模块的权重在"qkv_proj"模块中的"k"部分被共享，依此类推；这种权重共享可以减少模型的参数数量，并且在某些情况下可以提高模型的性能。
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config) # 包含了嵌入层和多个解码器层
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size) 
        if config.tie_word_embeddings: # 如果配置中指定了tie_word_embeddings为True，那么就将语言建模头的权重与嵌入层的权重共享，这样可以减少模型的参数数量，并且在某些情况下可以提高模型的性能。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward( # 定义了输入数据如何通过模型进行前向传播；输入包括input_ids和positions，分别表示输入的token ids和它们对应的位置；输出是模型的隐藏状态，可以用来计算logits或者进行其他任务。
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions) 

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
