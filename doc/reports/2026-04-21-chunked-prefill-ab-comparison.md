# Chunked Prefill 推广：Old vs New A/B 对比报告

- **日期**：2026-04-21
- **作者**：Orange-Bei（与 Claude 协作）
- **改动 commit**：`0a9c6da`（scheduler 一行删除 + spec/plan/测试）
- **相关 spec**：`doc/specs/2026-04-21-chunked-prefill-generalization-design.md`
- **复现资料**：`temp/chunked-prefill-ab/`（A/B 脚本 + 4 组 metrics JSON）

---

## 1. 背景

Nano-vLLM 的 `Scheduler.schedule` 在 prefill 循环里原有一条限制：**一步调度内只允许第一条 seq 被 chunk**。一旦已有至少一条 seq 进入 `scheduled_seqs`，后续任何 seq 若超本步剩余 budget 就直接 break，不被纳入，等下一步。

效果上：混合长短 prefill 负载时，每步的 `num_batched_tokens` 不会精确打满 `max_num_batched_tokens`，总有几百到几千 token 的 idle budget。本次改动（commit `0a9c6da`）把这条限制去掉，让任意 seq 都可以被 chunk，把 budget 用满。

本报告目的：**用数据回答"这个改动到底带来多少实际收益、在什么场景下明显、什么场景下不明显"**。

---

## 2. 代码差异

改动集中在 `nanovllm/engine/scheduler.py::schedule` prefill 循环里，共删除 3 行：

```diff
 while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
     seq = self.waiting[0]
     num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
     remaining = self.max_num_batched_tokens - num_batched_tokens
     if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
         break
-    if remaining < num_tokens and scheduled_seqs:    # no budget
-        # only allow chunked prefill for the first seq
-        break
     if not seq.block_table:
         self.block_manager.allocate(seq)
     ...
     seq.num_scheduled_tokens = min(num_tokens, remaining)
```

**语义变化：**

| 行为 | OLD | NEW |
|---|---|---|
| 第一条 seq 超 budget | 被 chunk（`min(num_tokens, remaining)`） | 被 chunk（同） |
| 后续任一 seq 超剩余 budget | **break 退出循环**，本步不纳入 | **被 chunk 进本步**，剩余 token 下一步继续 |
| 本步 `num_batched_tokens` 上限 | < `max_num_batched_tokens`（通常留白） | **= `max_num_batched_tokens`**（可打满） |

`min(num_tokens, remaining)` 这一行**原本就是 chunk 实现**，只是之前被上面的 `break` 挡住了对"非首条 seq"的路径。删掉 break 即激活。

---

## 3. 实验设计

### 3.1 对照原则

- **同 seed、同 prompts、同 sampling_params**：`random.seed(0)` 固定，两次跑的 256 条 prompt token_ids 与 `max_tokens` 完全一致
- **只切 scheduler.py**：A/B 之间只通过 `git checkout HEAD~1 -- nanovllm/engine/scheduler.py` 切换行为，其他代码、Python 环境、硬件（单卡 4090）全不变
- **排除冷启动噪声**：每次运行 bench 前先用 `["Benchmark: "]` warmup 一次，然后 `reset_metrics()`

### 3.2 两组配置

受益程度严重依赖 `max_num_batched_tokens` 相对 prompt 长度的比例，所以设计两组：

| 组 | `max_num_batched_tokens` | `max_model_len` | 预期效果 | 脚本 |
|---|---|---|---|---|
| G1（默认 bench） | 16384 | 4096 | 差异应不明显 | `bench.py` |
| G2（小 budget） | 2048 | 2048 | 差异应显著 | `temp/chunked-prefill-ab/bench_ab.py` |

负载：256 条 seq，input 长度 [100, 1024] 均匀随机，output 长度 G1 [100, 1024] / G2 [50, 256]（G2 压小 output 是为了让总耗时短一点，不影响 prefill 阶段的对比）。

### 3.3 记录的指标

- **Wall clock throughput**（tok/s）：总吞吐，最终用户可感
- **Prefill throughput**（tok/s）：改动直接影响的阶段
- **Decode throughput**（tok/s）：改动不触及，应保持不变（作对照组）
- **TTFT avg / p50 / p90**：首 token 延迟
- **Prefill step 数 + `num_batched_tokens` 分布**：直接反映 budget 填充率

---

## 4. 结果

### 4.1 G2：小 budget（`max_num_batched_tokens=2048`）——改动显效

**端到端吞吐与延迟**

| 指标 | OLD | NEW | Δ |
|---|---|---|---|
| Wall clock throughput | 4627 tok/s | **4820 tok/s** | **+4.2%** |
| Prefill throughput | 56878 tok/s | **64915 tok/s** | **+14.1%** |
| Decode throughput | 6800 tok/s | 6806 tok/s | +0.1%（持平） |
| 总时间 | 7.84 s | 7.52 s | −4.1% |
| TTFT avg | 1.503 s | **1.354 s** | **−9.9%** |
| TTFT p50 | 1.527 s | 1.353 s | −11.4% |
| TTFT p90 | 2.297 s | 2.009 s | −12.5% |

**Prefill step 分布（最本质的差异）**

| 指标 | OLD | NEW |
|---|---|---|
| Prefill step 数 | 84 | **70** |
| 平均 `num_batched_tokens` | 1700 | **2040** |
| 中位数 | 1724 | 2048 |
| Budget 填充率 | 83.0% | **99.6%** |
| 打满 budget（==2048）的步数 | **0 / 84** | **69 / 70** |

**解读：**

- OLD 有 84 个 prefill step、**每一个** 都留白（因为每步装到 budget-1 就 break 下一条）；NEW 只要 70 个 step，**几乎每一个** 都精确打满 2048
- 总 prefill 工作量不变（同样的 256 条 seq、同样的总 token 量），只是 OLD 把它摊到更多步、每步利用率低；NEW 压缩到更少步、每步利用率高
- Prefill 阶段加速了 14.1%，拖着 TTFT 同步降了 10~12%
- Decode 吞吐零变化，证明改动确实只作用于 prefill 阶段，没有副作用

### 4.2 G1：默认 budget（`max_num_batched_tokens=16384`）——改动几乎不可见

**端到端吞吐与延迟**

| 指标 | OLD | NEW | Δ |
|---|---|---|---|
| Wall clock throughput | 5279 tok/s | 5270 tok/s | −0.2%（噪声） |
| Prefill throughput | 92552 tok/s | 91642 tok/s | −1.0%（噪声） |
| Decode throughput | 5631 tok/s | 5624 tok/s | 持平 |
| TTFT avg | 1.016 s | 1.027 s | +1.1%（噪声） |

**Prefill step 分布**

| 指标 | OLD | NEW |
|---|---|---|
| Prefill step 数 | 9 | 9 |
| 平均 `num_batched_tokens` | 15870 | 15870 |
| 中位数 | 16114 | **16384** |
| 最大值 | 16364 | **16384** |
| 打满 budget（==16384）的步数 | **0 / 9** | **8 / 9** |

**解读：**

这是一个有意思的"质变但无收益"的场景：

- 填充率本质上**已经改善了**——OLD 从 0/9 打满升到 NEW 的 8/9 打满，median 从 16114 提升到 16384（精确打满）
- 但总 step 数相同（9）、总吞吐无变化——说明 OLD 的留白本就很小（16384 里留几百 token），chunk 能捡的边角料对总耗时无显著贡献
- 根本原因：256 条 prompt 的总量只有 ≈143K token，16384 budget 9 步就能装完；每步平均装 ≈29 条 seq，最后那条装不下的概率和留白占比都低

### 4.3 横向对比

| | G1 (budget=16384) | G2 (budget=2048) |
|---|---|---|
| Budget / max prompt 比 | 16× | 2× |
| OLD 填充率 | 96.9% | 83.0% |
| NEW 填充率 | 100%（8/9 打满） | 99.6% |
| Prefill 吞吐提升 | −1%（噪声） | **+14.1%** |
| 总吞吐提升 | −0.2%（噪声） | **+4.2%** |
| TTFT 改善 | 无 | **10~12%** |

---

## 5. 结论与应用建议

### 5.1 改动是否有效

**有效，但收益与配置强相关。**

- 当 `max_num_batched_tokens` 远大于单条 prompt 最大长度（≥8~16 倍）时，OLD 的 budget 留白本就很小，改动只是"把边角料填齐"，对端到端几乎无影响
- 当 `max_num_batched_tokens` 接近 prompt 最大长度（2~4 倍）时，OLD 的留白占 budget 显著比例，改动可拿到 10~15% 的 prefill 吞吐提升和 10%+ 的 TTFT 改善

### 5.2 何时会受益

生产配置里以下情形改动更显眼：

1. **显存紧张**：KV cache 挤压 `max_num_batched_tokens`，不得不设小
2. **长 prompt + 长 context**：`max_model_len` 长、prompt 接近 `max_num_batched_tokens`
3. **混合长短负载**：每步都有"最后一条装不下"的机会；纯长或纯短都不易触发
4. **低延迟敏感场景**：即使吞吐改善有限，TTFT 10% 级别的降低对交互式应用有感

### 5.3 何时无感

- 高并发、短 prompt、大 budget（典型聊天机器人稳态）：9 步打完 256 seq 的场景，改动对总时间无影响
- 单条超长 prompt（单 seq chunk）：OLD 本来就支持"第一条 seq chunk"，这个路径新旧等价

### 5.4 副作用评估

- **代码零风险**：删 3 行，保持所有原有不变量（spec §4.2 已枚举）
- **Decode 阶段零影响**：两组实验 decode throughput 持平
- **调度器开销**：每步多走 1~2 次循环迭代（纳秒级），可忽略
- **TTFT 边缘个案**：被 chunk 的某条 seq 可能比 OLD 多等一步才拿到 first token；但从 G2 数据看，统计意义上 TTFT 是改善的（长 prompt 被 chunk 的损失 < 后续 seq 更快被纳入的收益）

---

## 6. 复现步骤

所有实验材料在 `temp/chunked-prefill-ab/`：

- `bench_ab.py` — G2 的小 budget bench 脚本
- `bench_metrics_new_2048.json` / `bench_metrics_old_2048.json` — G2 的 raw metrics
- `bench_metrics_new_16384.json` / `bench_metrics_old_16384.json` — G1 的 raw metrics

**G1（默认 bench）：**

```bash
# NEW
python bench.py  # 写 temp/bench-runs/bench_metrics.json

# OLD
git checkout HEAD~1 -- nanovllm/engine/scheduler.py
python bench.py
git checkout HEAD -- nanovllm/engine/scheduler.py
```

**G2（小 budget bench）：**

```bash
# NEW
python temp/chunked-prefill-ab/bench_ab.py temp/chunked-prefill-ab/bench_metrics_new_2048.json

# OLD
git checkout HEAD~1 -- nanovllm/engine/scheduler.py
python temp/chunked-prefill-ab/bench_ab.py temp/chunked-prefill-ab/bench_metrics_old_2048.json
git checkout HEAD -- nanovllm/engine/scheduler.py
```

**步分布分析：**

```python
import json, statistics
d = json.load(open("temp/chunked-prefill-ab/bench_metrics_new_2048.json"))
prefills = [s["num_batched_tokens"] for s in d["step_samples"] if s["is_prefill"]]
print(f"n={len(prefills)}  mean={statistics.mean(prefills):.0f}  full={sum(1 for x in prefills if x == 2048)}")
```
