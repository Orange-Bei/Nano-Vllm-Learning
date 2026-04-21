# Nano-vLLM 一周学习计划

## 学习目标

这一周的目标不是只会跑通 `nano-vllm`，而是建立一套完整的推理框架认知。学完之后，你应该能够：

- 说清楚这个项目的整体结构，以及一次生成请求是如何从 prompt 走到 output 的。
- 理解 `prefill`、`decode`、`scheduler`、`KV cache`、`prefix cache`、`tensor parallel`、`CUDA Graph` 这些概念在代码中的落点。
- 区分“模型结构”与“推理框架能力”分别解决什么问题。
- 站在通用推理框架的视角，理解为什么 vLLM 类框架会比 naive 的逐条推理更高效。

## 代码地图

这份仓库的主链路很清晰：外部调用从 [README.md](../README.md) 里的示例和 [example.py](../example.py) 进入，实际入口是 [LLMEngine.generate](../nanovllm/engine/llm_engine.py)。生成请求先被封装成 `Sequence`，再交给 `Scheduler` 做 `prefill/decode` 调度，由 `BlockManager` 管理 KV cache block 和 prefix cache，接着 `ModelRunner` 负责准备张量、执行模型、调用 CUDA Graph 或 eager 路径，底层模型由 [Qwen3](../nanovllm/models/qwen3.py) 和 `layers/` 里的 attention、norm、linear、rope 等模块组成，最后由 [Sampler](../nanovllm/layers/sampler.py) 采样出 token 返回给上层。

建议你先记住这条主线：

`LLM.generate -> Scheduler -> BlockManager -> ModelRunner -> Qwen3/layers -> Sampler`

## Day 1: 建立全景图

### 学习目标

- 建立对仓库目录、核心模块和执行入口的整体认识。
- 先搞清楚“代码从哪里开始，结果从哪里出来”，不要一开始陷入底层细节。
- 画出一张 `generate` 主流程图，作为后续几天的总导航。

### 必读文件

- [README.md](../README.md)
- [example.py](../example.py)
- [bench.py](../bench.py)
- [nanovllm/llm.py](../nanovllm/llm.py)
- [nanovllm/engine/llm_engine.py](../nanovllm/engine/llm_engine.py)
- [nanovllm/config.py](../nanovllm/config.py)
- [nanovllm/sampling_params.py](../nanovllm/sampling_params.py)

### 关键问题

- 用户调用 `LLM.generate(...)` 后，程序依次经过了哪些模块？
- `LLM` 为什么几乎是空壳？真正的逻辑为什么都集中在 `LLMEngine`？
- `Config` 和 `SamplingParams` 分别控制什么？
- `example.py` 和 `bench.py` 分别代表“功能示例”和“性能路径”的哪两种典型使用方式？

### 动手任务

- 从 `example.py` 开始，手动顺着调用链追到 `LLMEngine.generate`。
- 把 `generate -> add_request -> step -> scheduler -> model_runner` 这条链写成一张简图。
- 记录每个顶层目录的职责：`engine/`、`layers/`、`models/`、`utils/`。
- 阅读 `bench.py`，理解为什么 benchmark 会先做一次 warmup。

### 当日验收

- 你能不看代码，口头复述一次 `generate` 的主流程。
- 你能解释 `example.py` 和 `bench.py` 分别想验证什么。
- 你手里已经有第一版“代码地图”和模块职责表。

## Day 2: 理解请求抽象与调度器

### 学习目标

- 理解一次请求在框架内部如何被表示、排队、运行、结束。
- 搞懂 `prefill` 与 `decode` 为什么要分开处理。
- 形成对 continuous batching 的第一层直觉。

### 必读文件

- [nanovllm/engine/sequence.py](../nanovllm/engine/sequence.py)
- [nanovllm/engine/scheduler.py](../nanovllm/engine/scheduler.py)
- [nanovllm/sampling_params.py](../nanovllm/sampling_params.py)

### 关键问题

- `Sequence` 里为什么既保存 `prompt` 信息，又保存运行期状态？
- `WAITING`、`RUNNING`、`FINISHED` 三种状态分别在什么时候切换？
- `prefill` 阶段为什么允许 `chunked prefill`，而 `decode` 阶段是按每次一个 token 调度？
- 什么情况下会发生 `preempt`，为什么要把正在运行的请求重新打回 waiting 队列？

### 动手任务

- 手动模拟 3 个不同长度请求进入 `Scheduler` 的过程。
- 画出 `waiting` 和 `running` 两个队列在一次调度循环中的变化。
- 标记 `num_cached_tokens`、`num_scheduled_tokens`、`num_completion_tokens` 这几个字段分别在什么阶段变化。
- 尝试回答：如果没有 `Scheduler`，直接逐条请求串行生成，吞吐为什么会差很多？

### 当日验收

- 你能解释 `schedule()` 为什么总是先尝试 `prefill`，再处理 `decode`。
- 你能说清楚 `preempt` 的触发条件和目的。
- 你能用一段自己的话解释 continuous batching 的核心思想。

## Day 3: 理解 KV Cache 与 Prefix Cache

### 学习目标

- 理解 `BlockManager` 是这个项目最关键的“显存管理层”之一。
- 搞懂 paged KV cache 的 block 视角，而不是只停留在“KV cache 是为了加速”这种口号层面。
- 理解 prefix cache 如何复用前缀，为什么要用 hash。

### 必读文件

- [nanovllm/engine/block_manager.py](../nanovllm/engine/block_manager.py)
- [nanovllm/engine/sequence.py](../nanovllm/engine/sequence.py)
- [nanovllm/engine/scheduler.py](../nanovllm/engine/scheduler.py)

### 关键问题

- `block_table` 为什么是序列与 KV cache 之间最重要的桥梁？
- `free_block_ids`、`used_block_ids`、`ref_count` 分别描述了什么资源状态？
- 为什么完整 block 才适合参与 prefix cache 复用？
- `hash_to_block_id` 为什么不能只依赖 hash，还要再校验 `token_ids`？
- `can_allocate`、`can_append`、`may_append` 三个动作分别对应什么运行阶段？

### 动手任务

- 手工画一个包含 2 到 3 个序列的 block 分配示意图。
- 模拟一个共享前缀的场景，观察第二个请求如何命中 prefix cache。
- 模拟一个序列结束后的 block 释放过程，理解 `ref_count` 为什么要倒序递减。
- 结合 `schedule()` 重新看 `block_manager.can_append(seq)`，理解 decode 阶段的显存压力点。

### 当日验收

- 你能画出 `Sequence.block_table` 与物理 block 之间的映射关系。
- 你能解释 prefix cache 命中时，为什么 `num_cached_tokens` 会增加。
- 你能说明这个实现为什么接近 vLLM 的 paged attention 思想。

## Day 4: 理解执行器与运行时上下文

### 学习目标

- 理解 `ModelRunner` 如何把调度器产出的逻辑请求转换成真正可执行的张量。
- 分清 `prefill` 路径和 `decode` 路径在输入组织上的根本区别。
- 理解 `Context` 为什么被设计成全局运行时上下文。

### 必读文件

- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)
- [nanovllm/utils/context.py](../nanovllm/utils/context.py)
- [nanovllm/config.py](../nanovllm/config.py)

### 关键问题

- `warmup_model()` 预热了什么，为什么这一步会影响后续显存估算？
- `allocate_kv_cache()` 是如何根据显存估算可用 block 数的？
- `prepare_prefill()` 里为什么要构造 `input_ids`、`positions`、`cu_seqlens_q`、`cu_seqlens_k`、`slot_mapping`？
- `prepare_decode()` 为什么只喂每个序列的 `last_token`？
- `Context` 里保存的数据为什么能支撑 attention 层读写 KV cache？

### 动手任务

- 逐行梳理 `prepare_prefill()` 和 `prepare_decode()`，分别写出每个输出张量的语义。
- 为一个简单 batch 手动推演 `slot_mapping` 是如何映射到 cache slot 的。
- 计算 `block_bytes` 的公式分别由哪些维度组成。
- 标记 `run_model()` 在什么情况下走 eager，什么情况下走 CUDA Graph。

### 当日验收

- 你能说清楚 `prefill` 和 `decode` 输入组织的不同。
- 你能解释 `slot_mapping` 与 block cache 写入之间的关系。
- 你能概括 `ModelRunner` 在整个框架中的职责边界。

## Day 5: 理解模型结构与注意力路径

### 学习目标

- 理解模型本身如何接入推理框架，而不是把注意力只当作纯数学公式。
- 搞懂 `Qwen3` 这一层代码哪些是模型特性，哪些是推理优化的接口。
- 建立从 `input_ids` 到 `logits` 的前向视角。

### 必读文件

- [nanovllm/models/qwen3.py](../nanovllm/models/qwen3.py)
- [nanovllm/layers/attention.py](../nanovllm/layers/attention.py)
- [nanovllm/layers/layernorm.py](../nanovllm/layers/layernorm.py)
- [nanovllm/layers/rotary_embedding.py](../nanovllm/layers/rotary_embedding.py)
- [nanovllm/layers/activation.py](../nanovllm/layers/activation.py)

### 关键问题

- 一层 decoder block 中 attention、RMSNorm、MLP 的顺序是什么？
- `Qwen3Attention` 中 `q/k/v` 是如何切分与 reshape 的？
- `Attention.forward()` 为什么要先写 cache，再调用 flash attention？
- `flash_attn_varlen_func` 与 `flash_attn_with_kvcache` 分别对应什么阶段？
- `ParallelLMHead` 为什么在 prefill 阶段只取每个序列最后一个位置的 hidden states？

### 动手任务

- 画出一层 `Qwen3DecoderLayer` 的前向图。
- 手动追踪 `Qwen3ForCausalLM.forward -> model -> layers -> norm -> lm_head`。
- 对照 `attention.py` 和 `context.py`，写下 attention 层依赖了哪些运行时信息。
- 尝试区分哪些模块是“模型实现”，哪些模块更接近“推理框架接口”。

### 当日验收

- 你能从代码层面解释一次 attention 是如何读写 KV cache 的。
- 你能说明 `Qwen3` 只是当前模型实现，而不是整个框架的核心抽象。
- 你能讲清楚 hidden states 最后是怎样变成 logits 的。

## Day 6: 理解并行、权重装载与系统优化

### 学习目标

- 理解这个项目如何用尽量少的代码表达 tensor parallel。
- 理解权重是如何按 rank 切分并装载的。
- 把“模型计算”与“分布式运行机制”真正连接起来。

### 必读文件

- [nanovllm/layers/linear.py](../nanovllm/layers/linear.py)
- [nanovllm/layers/embed_head.py](../nanovllm/layers/embed_head.py)
- [nanovllm/utils/loader.py](../nanovllm/utils/loader.py)
- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)

### 关键问题

- `ColumnParallelLinear` 与 `RowParallelLinear` 的切分维度为什么不同？
- `QKVParallelLinear` 和 `MergedColumnParallelLinear` 分别为哪类融合权重服务？
- `VocabParallelEmbedding` 和 `ParallelLMHead` 是如何处理词表切分的？
- `dist.all_reduce` 和 `dist.gather` 在这里分别承担什么角色？
- 多进程 rank 之间为什么通过共享内存和事件同步来驱动执行？

### 动手任务

- 结合 `weight_loader` 逻辑，手工推演一个参数在不同 rank 上如何切片。
- 画出 rank 0 与其他 rank 的职责差异。
- 阅读 `ModelRunner.__init__` 和 `loop()`，理解为什么 rank 0 是控制面，其他 rank 更像执行面。
- 总结一张“算子并行/词表并行/权重装载”的对照表。

### 当日验收

- 你能说明列并行、行并行、词表并行各自的切分方式和通信方式。
- 你能解释 `loader.py` 为什么需要 `packed_modules_mapping`。
- 你能说清楚多卡情况下，这个项目最小可行的执行控制流。

## Day 7: 从项目回到推理框架视角

### 学习目标

- 把过去六天的代码级理解上升为推理框架级理解。
- 建立“请求层、调度层、显存层、执行层、并行层、模型层”的整体认知图。
- 输出一份自己的总结，而不是停留在被动阅读状态。

### 必读文件

- [nanovllm/engine/llm_engine.py](../nanovllm/engine/llm_engine.py)
- [nanovllm/engine/scheduler.py](../nanovllm/engine/scheduler.py)
- [nanovllm/engine/block_manager.py](../nanovllm/engine/block_manager.py)
- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)
- [nanovllm/layers/attention.py](../nanovllm/layers/attention.py)
- [nanovllm/layers/sampler.py](../nanovllm/layers/sampler.py)

### 关键问题

- 如果去掉 `Scheduler`、`BlockManager`、`CUDA Graph`、`prefix cache`，系统会分别退化成什么样？
- 为什么说推理框架的价值主要体现在“如何调度与复用资源”，而不是只体现在“模型能不能跑”？
- `nano-vllm` 哪些设计是通用的，哪些是为了当前实现做的简化？
- 这个项目与更完整的 vLLM 相比，还缺少哪些能力？

### 动手任务

- 用一页纸总结这个项目的完整数据流。
- 按“请求层、调度层、显存层、执行层、并行层、模型层”六个视角写总结。
- 回答一个核心问题：为什么 vLLM 类推理框架往往比 naive 推理更高吞吐？
- 如果你有运行环境，执行 [bench.py](../bench.py) 并记录吞吐；如果没有运行环境，就根据代码推断吞吐优化来源。

### 当日验收

- 你能独立讲解这个项目的推理链路，不需要再跟着代码逐行看。
- 你能用框架视角而不是模型视角介绍 `nano-vllm`。
- 你已经形成一份自己的总结笔记，可以作为后续深入 vLLM、SGLang、TensorRT-LLM 的出发点。

## 每天固定动作

- 先读代码 60 到 90 分钟，再用 30 分钟整理笔记。
- 每天都回答三个固定问题：这个模块解决什么问题、核心数据结构是什么、如果去掉它会损失什么。
- 每天尽量做一个小实验。能运行就跑一次，不能运行就手推数据流和状态变化。
- 每天结束时，把当天的理解压缩成 5 到 10 句话，避免“看懂了但讲不出来”。

## 推理框架认知图

### 请求层

- 负责接收 prompt、采样参数、序列状态。
- 在这个仓库里主要由 [nanovllm/engine/sequence.py](../nanovllm/engine/sequence.py) 和 [nanovllm/sampling_params.py](../nanovllm/sampling_params.py) 承担。

### 调度层

- 负责决定哪些请求先 `prefill`，哪些请求进入 `decode`，以及何时抢占。
- 在这个仓库里主要由 [nanovllm/engine/scheduler.py](../nanovllm/engine/scheduler.py) 承担。

### 显存层

- 负责以 block 为单位管理 KV cache、复用 prefix、控制序列与物理 cache 的映射。
- 在这个仓库里主要由 [nanovllm/engine/block_manager.py](../nanovllm/engine/block_manager.py) 承担。

### 执行层

- 负责把逻辑请求转成 GPU 上的张量执行，组织 `prefill/decode` 输入，并调度 eager 或 CUDA Graph。
- 在这个仓库里主要由 [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py) 和 [nanovllm/utils/context.py](../nanovllm/utils/context.py) 承担。

### 并行层

- 负责多卡张量并行、词表切分、进程通信和权重分发。
- 在这个仓库里主要由 [nanovllm/layers/linear.py](../nanovllm/layers/linear.py)、[nanovllm/layers/embed_head.py](../nanovllm/layers/embed_head.py)、[nanovllm/utils/loader.py](../nanovllm/utils/loader.py) 承担。

### 模型层

- 负责定义网络结构本身，包括 attention、MLP、RMSNorm、RoPE 和 logits 头。
- 在这个仓库里主要由 [nanovllm/models/qwen3.py](../nanovllm/models/qwen3.py) 和 `layers/` 下相关模块承担。

## 最后提醒

学习这个项目时，不要一开始把主要精力花在 `Qwen3` 结构细节上。对“推理框架”真正有迁移价值的，是 `engine/` 和 `layers/` 里那些围绕调度、缓存、执行和并行展开的设计。模型会换，但这些问题会反复出现。
