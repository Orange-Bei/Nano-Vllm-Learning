# Day 3-4 学习笔记：模型层与 layers/ 各模块

> 本两天啃下 `models/qwen3.py` + `layers/` 全部 7 个文件 + `utils/loader.py`，把 Qwen3 架构、TP（张量并行）和权重装载串成一张完整图。

---

## 0. 学习地图：layers/ 下 8 个文件的三桶分类

```
nanovllm/
├─ models/qwen3.py            # 模型骨架（Qwen3-specific）
├─ layers/
│  ├─ linear.py               # ★ TP 并行 Linear（4 个类）
│  ├─ embed_head.py           # ★ TP 并行 Embedding / LM Head
│  ├─ attention.py            # paged KV + flash-attn
│  ├─ rotary_embedding.py     # RoPE
│  ├─ layernorm.py            # RMSNorm（含 add_rms 融合版）
│  ├─ activation.py           # SiluAndMul (SwiGLU)
│  └─ sampler.py              # Gumbel-max 采样
└─ utils/loader.py            # safetensors → 并行权重分片
```

| 桶 | 文件 | 解决的问题 |
|----|------|-----------|
| **模型实现** | `qwen3.py`, `layernorm.py`, `activation.py`, `rotary_embedding.py` | Qwen3 架构定义 + 纯数学算子 |
| **并行封装** | `linear.py`, `embed_head.py`, `loader.py` | TP 切分 + 装载 |
| **推理优化** | `attention.py`, `sampler.py` | paged KV + Gumbel-max |

**关键认识**：TP=1 时，"并行封装"桶里的类全部退化成普通 `nn.Linear` / `nn.Embedding`。这些类存在的唯一理由是多卡，换句话说**模型正确性不依赖它们**。

---

## 1. Qwen3 模型层（`qwen3.py`）

### 1.1 四层嵌套结构

```
Qwen3ForCausalLM                            # 最外层：模型 + lm_head
  └─ Qwen3Model                             # 主干：embed + N 层 decoder + 最终 norm
       ├─ VocabParallelEmbedding            # embed_tokens
       ├─ Qwen3DecoderLayer × num_layers    # 每层
       │    ├─ RMSNorm (input_layernorm)
       │    ├─ Qwen3Attention
       │    │    ├─ QKVParallelLinear
       │    │    ├─ RMSNorm (q_norm, k_norm)     ← Qwen3 独有 QK-Norm
       │    │    ├─ RotaryEmbedding
       │    │    ├─ Attention (paged KV + flash-attn)
       │    │    └─ RowParallelLinear (o_proj)
       │    ├─ RMSNorm (post_attention_layernorm)
       │    └─ Qwen3MLP
       │         ├─ MergedColumnParallelLinear (gate_up_proj)
       │         ├─ SiluAndMul
       │         └─ RowParallelLinear (down_proj)
       └─ RMSNorm (final norm)
  └─ ParallelLMHead
```

### 1.2 Residual 的"接力棒"模式

不走常见的 `h = h + attn(ln(h))`，而是把"加法"和"下一次 norm"**融合**进 `add_rms_forward`：

```python
def forward(self, positions, hidden_states, residual):
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

- `add_rms_forward(x, residual)` 内部做 `x = x + residual; residual = x; norm(x)`——加法和归一化发生在同一个 `@torch.compile` 块里。
- 每层交出去的 residual = "该层输入 + attn_out"（还差 mlp_out 没加）。
- 第一层 residual=None 时走 `rms_forward` 初始化。
- 最后的 `self.norm(hidden, residual)` 把最后一层的 mlp_out 和 residual 合并，补齐"最后一层 MLP 的贡献"。

### 1.3 四个关键设计决策

1. **Residual 作为参数进出** — 融合加法 + norm，每层省一次 kernel。
2. **Q/K 独有 RMSNorm (QK-Norm)** — Qwen3 模型特性，不是框架能力；稳住深层训练 softmax 数值。
3. **`packed_modules_mapping`** — 让 HF checkpoint 的散件（q_proj、k_proj、v_proj）装进融合权重矩阵（qkv_proj）。
4. **`tie_word_embeddings`** — 小模型 `lm_head.weight.data = embed_tokens.weight.data` **浅拷贝**共享内存，省一份 vocab×hidden 的参数。Qwen3-0.6B 靠这个省 ~300MB。

---

## 2. TP 的数学基础

### 2.1 切权重两种方向

`y = x @ W`，W shape `[in, out]`：

```
Column（切输出维）                   Row（切输入维）
W = [W₀ | W₁ | ... | W_k]            W = [W₀]
                                         [W₁]
                                         [...]
                                         [W_k]

每卡需要完整 x                       每卡只拿 x 的一段
每卡输出完整 y 的几列                 每卡输出完整 y 的部分和
结果 concat 即可（无通信）             需要 all_reduce 求和
```

### 2.2 具体数值证明（TP=2, x shape=[1,4], W=[4,4]）

**Column 切**：每 yⱼ 只依赖 W 的第 j 列——这一列只在一张卡上，自然独立产出完整值。
**Row 切**：yⱼ = Σᵢ xᵢ·wᵢⱼ，i 拆到不同卡只算部分和，必须 all_reduce 汇总。

### 2.3 Column→Row 配对的妙处

一层 decoder 的两对 Column→Row：

```
Attention: hidden ─► qkv_proj(Col) ─► split ─► attention ─► o_proj(Row)    ─► hidden'
                                                              [all_reduce]
MLP:       hidden ─► gate_up_proj(Col) ─► silu·mul ─► down_proj(Row)        ─► hidden'
                                                              [all_reduce]
```

- Column 输出已切分，正是 Row 输入所需的切片形态。
- 中间 attention/silu·mul **零通信**。
- 一层只有 **2 次 all_reduce**（attention 出口 + MLP 出口），和 TP 规模无关。

### 2.4 x 在一层内的完整/切分状态

```
层入口：   hidden 完整（每卡一份）
qkv_proj： 输入完整 → 输出切分
attention：切分 → 切分（head 独立）
o_proj：   切分 → 部分和 → all_reduce → 完整
ln + mlp 入口：hidden 完整
gate_up：  完整 → 切分
silu·mul： 切分 → 切分
down_proj：切分 → 部分和 → all_reduce → 完整
层出口：   hidden 完整
```

**完整态只出现在层入口/出口；中间态全是切分的**。

---

## 3. `linear.py` 四个并行类

### 3.1 LinearBase 骨架

- `weight` shape `[output, input]`；`tp_dim=0` 切输出，`tp_dim=1` 切输入。
- `weight.weight_loader = self.weight_loader` — **让 Parameter 自己知道怎么加载自己**，loader 只做名字路由。

### 3.2 四个类对照

| 类 | tp_dim | 每卡 weight shape | forward 通信 | 用途 |
|----|--------|------------------|------------|------|
| ColumnParallelLinear | 0 | `[out/TP, in]` | 无 | 基础 Column |
| RowParallelLinear | 1 | `[out, in/TP]` | `all_reduce` | o_proj, down_proj |
| QKVParallelLinear | 0 | `[(nH+2·nKV)·hd/TP, hidden]` | 无（同 Col） | 三合一的 QKV |
| MergedColumnParallelLinear | 0 | `[sum(out_sizes)/TP, in]` | 无（同 Col） | 二合一的 gate_up |

### 3.3 QKVParallelLinear 的切分必须按 head 粒度

```python
output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
```

Qwen3-0.6B: `(16 + 2*8) * 128 = 4096`；TP=2 下每卡 2048 列，分三段：

| 段 | 起点 | 长度 | 内容 |
|---|------|------|------|
| Q | 0 | 8·128=1024 | Q 头 [0-7] 或 [8-15] |
| K | 1024 | 4·128=512 | K 头 [0-3] 或 [4-7] |
| V | 1536 | 4·128=512 | V 头 [0-3] 或 [4-7] |

**head 不跨卡**——GQA 正确性要求每个 Q head 和它分组内的 KV head 在同一张卡上。

### 3.4 RowParallelLinear 的 bias 处理

```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size > 1:
    dist.all_reduce(y)
```

**bias 只在 rank 0 加**，否则 all_reduce 一下 bias 被加 TP 次。

---

## 4. `embed_head.py`：首尾转换 + vocab 切分

### 4.1 VocabParallelEmbedding：id → vector

**权重切分**：每卡持有 `vocab_size/TP` 行 embedding，负责特定 id 区间。

**mask + all_reduce 技巧**：

```python
mask = (x >= vocab_start_idx) & (x < vocab_end_idx)
x = mask * (x - vocab_start_idx)           # 范围外映到 0
y = F.embedding(x, self.weight)            # 每卡查自己的表
y = mask.unsqueeze(1) * y                  # 范围外的结果置零
dist.all_reduce(y)                         # 汇总：每卡都拿到完整 embedding
```

范围外的查表浪费一次，但换来代码统一。

### 4.2 ParallelLMHead：vector → logits

继承 VocabParallelEmbedding（共享 weight_loader + 切分方式），但 forward 完全不同：

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()        # 只取每条 seq 末位
logits = F.linear(x, self.weight)           # [N, vocab/TP]
if tp_size > 1:
    dist.gather(logits, all_logits, 0)      # 汇到 rank 0
    logits = torch.cat(all_logits, -1) if rank==0 else None
```

两个优化：
- **prefill 只取末位**：`cu_seqlens_q[1:] - 1` 拿到每条 seq 最后一 token 的打平索引。采样只需末位 logits。计算量从 `N·vocab` 降到 `num_seqs·vocab`。
- **gather 而非 all_gather**：只有 rank 0 做采样，其他 rank 拿完整 logits 没用，省一半通信。

### 4.3 VocabParallelEmbedding 和 ParallelLMHead 对比

| 维度 | VocabParallelEmbedding | ParallelLMHead |
|------|----------------------|----------------|
| 方向 | id → vector | vector → logits |
| 操作 | 查表（F.embedding） | matmul（F.linear） |
| 通信 | all_reduce | gather 到 rank 0 |
| prefill 优化 | 无 | 只取末位 |

### 4.4 和 tie_word_embeddings 联动

`lm_head.weight.data = embed_tokens.weight.data` 浅拷贝：
- checkpoint 里只存 `embed_tokens.weight`，不存 `lm_head.weight`。
- 加载 embed 后 lm_head 自动就位（共享内存）。
- TP 下两个 module 的 vocab 切分方式完全一致（靠继承保证），每卡共享同一块 `[vocab/TP, hidden]`。

---

## 5. `attention.py` 的 TP 视角

### 5.1 Attention 模块对 TP 完全无感

```python
self.num_heads = total_num_heads // tp_size       # 每卡本地 head 数
self.num_kv_heads = total_num_kv_heads // tp_size
```

attention 本身夹在 Column→Row 之间，只处理本卡的 head，**不跨卡通信**。

### 5.2 Paged KV 也是每卡独立

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
self.kv_cache = torch.empty(2, num_hidden_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

每张卡有自己的 paged KV，存本卡负责的 kv heads。block_id 全局一致（调度器视角），物理内容各卡不同。

### 5.3 Context 是"逻辑索引"

`cu_seqlens / slot_mapping / block_tables` 描述 token 和 block 的关系，不描述 head。每卡拿同一份 Context，各自索引本卡 kv_cache。**TP 的切分藏在权重里，不在控制流里**。

---

## 6. `sampler.py`：Gumbel-max trick

```python
logits = logits.float().div_(temperatures.unsqueeze(1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)
```

### 6.1 数学等价

按分布 p 采样 ≡ `argmax(log p + G)`（G ~ Gumbel(0,1)）≡ `argmax(p / Exp(1))`。

证明：`-log(E) ~ Gumbel` 当 `E ~ Exp(1)`，所以
```
argmax(log p + G) = argmax(log p - log E) = argmax log(p/E) = argmax(p/E)
```

### 6.2 为什么不用 `torch.multinomial`

- multinomial 内部带不可 compile 的 RNG 状态推进，`@torch.compile` 跑不动或跑得慢。
- Gumbel-max 全是 elementwise + argmax，纯算术，能被 torch.compile 融成一个 kernel。
- GPU 并行友好：batch 中所有位置独立。

### 6.3 温度效果

- T→0：分布变尖，退化到 argmax（贪心）。
- T→∞：分布变平，退化到均匀采样。
- T=0 会除零——nano-vllm 没做保护，用户传 0 会 NaN。

---

## 7. `loader.py`：权重装载全流程

21 行代码，是模型冷启动的枢纽：

```python
for file in glob("*.safetensors"):
    for weight_name in f.keys():
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                param.weight_loader(param, f.get_tensor(weight_name), shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, f.get_tensor(weight_name))
```

### 7.1 两条路径

**路径 A：融合权重**（q/k/v/gate/up）
1. 匹配 `packed_modules_mapping` 里的 key。
2. 改写参数名：`q_proj.weight` → `qkv_proj.weight`。
3. 调用目标 param 的 `weight_loader(param, loaded_tensor, shard_id)`，内部按 shard_id 塞到融合矩阵的对应段 + TP 切分。

**路径 B：普通权重**（for-else 的 else）
- Python for-else：`else` 在 for 正常结束（没 break）时执行。
- 名字不改写，调自带 weight_loader（或 default_weight_loader 直接 copy）。

### 7.2 关键设计：weight_loader 挂在 Parameter 上

- loader **不需要判断**"这是 Column 还是 Row 还是 QKV"。
- Parameter 自己知道怎么加载——这是运行时多态分发。
- `packed_modules_mapping` 只管"名字路由"，具体切分由 `weight_loader` 实现。

### 7.3 TP 下每卡独立加载

每张卡都跑一遍 `load_model`，但 `weight_loader` 里 `loaded_weight.chunk(tp_size)[tp_rank]` 只取本 rank 切片，每卡只把自己那部分从 CPU 搬到 GPU。

### 7.4 tie_word_embeddings 的处理

Qwen3 checkpoint 里**没有 `lm_head.weight`**。loader 只会加载 embed_tokens，lm_head 通过浅拷贝自动就位。`get_parameter("lm_head.weight")` 永远不被触发，不会报 KeyError。

---

## 8. 次要算子（和 TP 无关）

这三个都是 elementwise / 沿 hidden 维的操作，hidden 每卡完整，独立算就对。

### 8.1 `rotary_embedding.py` — RoPE

- **作用**：让 attention 的 `Q·K^T` 只依赖 token 间相对位置（n - m），不依赖绝对位置。
- **数学**：把 Q/K 的 head_dim 两两配对看成 2D 向量，按位置 m 旋转角度 `m·θ`。`<R_m q, R_n k> = <q, R_{n-m} k>`。
- **实现**：预算 `[max_pos, head_dim]` 的 cos/sin 表（init 一次），每步 attention 前查表 + 2D 旋转公式 `y1=x1·cos - x2·sin, y2=x2·cos + x1·sin`。
- **不同频率**：`inv_freq = 1 / (base^(2i/d))`，前几对维度旋得快（短距离）、后几对慢（长距离）。Qwen3 base=1e6（LLaMA=1e4），频率更低、长序列外推更稳。
- **V 不旋转**——位置信息只需进入 QK 内积。
- **RoPE 在写 cache 前做**：cache 里存的 K 是旋转后的，decode 时直接读，复用安全。

### 8.2 `layernorm.py` — RMSNorm

```python
rms_forward(x):      # 单参数版
    var = x.pow(2).mean(-1, keepdim=True)
    return x * rsqrt(var + eps) * weight

add_rms_forward(x, residual):   # 融合版
    x = x + residual
    residual = x
    ... 同 rms ...
    return normed, residual
```

两个都 `@torch.compile` 融合。第一层走 `rms_forward`（没 residual 可加），后续走 `add_rms_forward`（融合加法 + norm）。

### 8.3 `activation.py` — SiluAndMul (SwiGLU)

```python
x, y = x.chunk(2, -1)
return F.silu(x) * y
```

MergedColumnParallelLinear 把 gate 和 up 拼在一起一次 matmul 出 `[2·intermediate]`，SiluAndMul 把它切两半、`silu(gate) * up`。标准 SwiGLU 激活。

---

## 9. 前向传播的控制流

### 9.1 ModelRunner.run() 三阶段

```
① prepare_prefill/decode  ─►  ② run_model（前向）  ─►  ③ sampler
   （CPU 拼张量 + 写 Context）    （从 self.model(input_ids, positions) 开始）   （仅 rank 0）
```

### 9.2 前向触发机制

PyTorch 里 `module(args)` 自动调 `__call__ → forward`，**不能直接调 `forward`**。

### 9.3 一次 prefill 的完整调用栈

```
run_model
└─ self.model(input_ids, positions)
   └─ Qwen3ForCausalLM.forward
      └─ Qwen3Model.forward
         ├─ embed_tokens(...) → VocabParallelEmbedding.forward [all_reduce]
         ├─ for layer in layers:  (N 次)
         │  └─ Qwen3DecoderLayer.forward
         │     ├─ input_layernorm → RMSNorm.forward
         │     ├─ self_attn → Qwen3Attention.forward
         │     │  ├─ qkv_proj → QKVParallelLinear.forward
         │     │  ├─ split + q_norm + k_norm
         │     │  ├─ rotary_emb → RotaryEmbedding.forward
         │     │  ├─ attn → Attention.forward (store_kvcache + flash_attn)
         │     │  └─ o_proj → RowParallelLinear.forward [all_reduce ①]
         │     ├─ post_attention_layernorm
         │     └─ mlp → Qwen3MLP.forward
         │        ├─ gate_up_proj → MergedColumnParallelLinear.forward
         │        ├─ act_fn → SiluAndMul.forward
         │        └─ down_proj → RowParallelLinear.forward [all_reduce ②]
         └─ norm → RMSNorm.forward
└─ compute_logits → lm_head → ParallelLMHead.forward [gather]
```

**前向是深度优先的树遍历**，每个 `module(x)` 就是一次进入，forward 返回就是退出。不是"并行调各类 forward"，是**递归嵌套**调用。

---

## 10. 通信原语速查

| 位置 | 原语 | 为什么 |
|------|------|-------|
| VocabParallelEmbedding 出口 | `all_reduce` | 每卡范围外置零，sum 后每卡都有完整 embedding |
| 每层 o_proj 出口 | `all_reduce` | Row 切输入 → 部分和 |
| 每层 down_proj 出口 | `all_reduce` | 同上 |
| ParallelLMHead 出口 | `gather` 到 rank 0 | 只有 rank 0 采样，省带宽 |

一层 decoder = 2 次 all_reduce；整个模型 = 1 次 embed all_reduce + 28·2 次层内 all_reduce + 1 次 gather（Qwen3-0.6B）。

---

## 11. 关键不变量

1. **TP 切的是权重**；hidden 在层入口/出口完整，中间态切分。
2. **Column 吃完整 x 输出切分**；**Row 吃切分 x 输出部分和需 all_reduce**。
3. **Column→Row 配对**中间零通信，一对贡献一次 all_reduce。
4. **Parameter 自带 weight_loader**，loader 只做名字路由。
5. **QKV/gate_up 融合矩阵**按 shard_id 分段装载，每段内再按 TP 切。
6. **QK-Norm、tie_word_embeddings、QKV bias=False、base=1e6** 都是 Qwen3 模型特性，换模型就换。
7. **RoPE 在写 KV cache 之前做**，cache 里的 K 是旋转后的。
8. **attention / RMSNorm / SiluAndMul / RoPE 对 TP 无感**——elementwise 或沿完整 hidden 维。

---

## 12. 压缩 5 句话

1. **qwen3.py 是架构图**，layers/ 是算子；换模型只换架构图，算子通用。
2. **TP 切的是权重**：Column 切输出（无通信）、Row 切输入（all_reduce），Column→Row 配对中间零通信、一对一次 all_reduce。
3. **融合权重（qkv/gate_up）靠 packed_modules_mapping + 参数自带的 weight_loader** 配合，loader 只是邮差。
4. **Embedding mask+all_reduce**（输入是 id，范围外置零后汇总），**LM head gather 到 rank 0**（只 rank 0 采样）+ prefill 时只取末位。
5. **前向是递归树遍历**：`self.model(x)` 按 PyTorch 的 `module(args) → forward` 约定级联触发，深度优先一路下到 flash-attn 和 Triton kernel，再逐层返回。

---

## 13. 三天学习回顾（Day 1-4）

- **Day 1**：全景图——`generate()` 主流程、Scheduler、ModelRunner、Attention、Sampler，怎么从 prompt 到 output。
- **Day 2**：BlockManager——paged KV 的分配/回收/prefix cache，块的完整生命周期。
- **Day 3-4**：模型层与算子——Qwen3 架构、TP 的 Column/Row 切分、融合权重装载、Embedding/LM Head 的通信原语、Sampler 数学、前向控制流。

下一步可以去看 `ModelRunner.__init__` 的多进程启动、CUDA Graph capture/replay、共享内存通信，把**运行时层**补齐——planday3 原本预告的 Day 5 内容。
