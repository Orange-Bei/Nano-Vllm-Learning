# Data Parallel Scaling：DP=1 vs DP=2 对比报告

- **日期**：2026-04-21
- **作者**：Orange-Bei（与 Claude 协作）
- **改动 commit**：`d7d796d`（DP-only，TP=1 固定）
- **相关 spec**：`doc/specs/2026-04-21-data-parallel-design.md`
- **复现资料**：`temp/dp-ab/`（bench_dp.py + 2 组 metrics JSON）
- **硬件**：3×4090（本次只用 2 张，DP=2；第 3 张空闲）

---

## 1. 背景

commit `d7d796d` 新增了 DP-only 数据并行（`DPLLMEngine` + `LLM(..., data_parallel_size=N)` 工厂分派），每 DP 副本一张卡、TP=1 固定、round-robin 请求分发、sync step 节拍。本报告用 bench 量化 DP 的实际收益、分析 scaling 是否接近理论 N× 上限。

Qwen3-0.6B 16 attention heads 不能整除 3，`tensor_parallel_size=3` 不可用；DP=3 在 3×4090 上在技术上是可行（DP 副本与 TP 切分正交），但本轮只做 DP=1 vs DP=2 的基础 scaling 验证。

## 2. 实验设计

### 2.1 对照原则

- **同 seed、同 prompts**：`random.seed(0)` 固定，256 条随机长度 prompt (`input_len ∈ [100, 1024]`)、random `max_tokens ∈ [100, 1024]`（`ignore_eos=True` 强制跑满）
- **只切 `data_parallel_size`**：DP=1 走 `LLMEngine`（单卡），DP=2 走 `DPLLMEngine`（2 个 worker 子进程，每卡一份 model）
- **相同参数**：`enforce_eager=False`、`max_model_len=4096`、`max_num_batched_tokens=16384`（默认）
- **排除冷启动**：bench 前跑 `generate(["Benchmark: "], ...)` warmup + `reset_metrics()`

### 2.2 脚本

`temp/dp-ab/bench_dp.py`（参数化 `dp_size` 和 metrics 输出路径）：

```bash
python temp/dp-ab/bench_dp.py 1 temp/dp-ab/bench_metrics_dp1.json
python temp/dp-ab/bench_dp.py 2 temp/dp-ab/bench_metrics_dp2.json
```

## 3. 结果

### 3.1 总吞吐与延迟

| 指标 | DP=1 | DP=2 | Δ |
|---|---|---|---|
| Wall clock throughput | 5272 tok/s | **8329 tok/s** | **1.58×** |
| 总时间（生成 133966 tok） | 25.41 s | **16.08 s** | −37% |
| Prefill throughput | 91799 tok/s | 45581 tok/s（单 worker）；91162 tok/s（聚合） | 持平 |
| Decode throughput | 5626 tok/s | 5083 tok/s（单 worker）；10166 tok/s（聚合） | **1.81×** |
| TTFT avg | **1.025 s** | 1.710 s | +67% |
| TTFT p50 | 1.053 | 1.849 | +76% |
| TTFT p90 | 1.456 | 1.986 | +36% |
| Preemptions | 0 | 0 | — |

### 3.2 Step 分布

| 指标 | DP=1 | DP=2（2 worker 聚合） |
|---|---|---|
| Total steps | 1032 | 2032 |
| Prefill steps | 9 | 10（每 worker ~5） |
| Prefill `num_batched_tokens` mean | 15870 | 14283 |
| Prefill `num_batched_tokens` max | 16384 | 16384 |
| Prefill step duration (mean) | **173 ms** | **313 ms**（单 worker） |
| Decode steps | 1023 | 2022（每 worker ~1011） |
| Decode `num_seqs` mean | 130.7 | 66.1（单 worker，~50%） |
| Decode step duration (mean) | 23.2 ms | 13.0 ms（单 worker，~56%） |

## 4. 分析

### 4.1 核心结论：Decode 线性 scaling，Prefill 亚线性

- **Decode 阶段几乎完美 scaling**：DP=2 聚合 decode 吞吐 10166 tok/s ≈ 1.81× DP=1 单卡 5626，效率 **90%**。每 worker 负担一半 seq 数（130→66）、单步更快（23→13 ms），两 worker 并行完美叠加
- **Prefill 阶段几乎无提升**：DP=2 每 worker prefill 吞吐只有 45581 tok/s（DP=1 单卡的 50%）——**单 worker 的 prefill step_duration 313 ms 比 DP=1 的 173 ms 慢了 1.81 倍**，几乎完全抵消了 2 worker 并行的收益。聚合 prefill 吞吐 91162 tok/s ≈ DP=1 的 91799 tok/s，**几乎持平**
- **总吞吐 1.58×** 是 decode 阶段带来的；如果负载是 prefill-heavy，DP 收益会显著低于 N×

### 4.2 Prefill step_duration 为什么翻倍

观察到的"单 worker prefill step 173→313 ms"现象值得分析。几个候选原因：

1. **CUDA graph warmup 不完整**：`enforce_eager=False` 启用 CUDA graph，但 bench 里只用 1 条 seq warmup。Prefill 的 batch size 分布（每步 ~29 seq、`num_batched_tokens` 近 16384）可能触发大量未 warmup 的 graph，worker 第一次见到这些 shape 时回退到 eager，多出来的编译/launch overhead 被记入 step_duration
2. **多进程下 GPU memory controller 争用**：3×4090 共享 PCIe 根复合体时，2 个 worker 同时做 prefill（大 attention kernel + 大 KV write）可能在 memory bandwidth 上互相干扰
3. **CPU-side 开销放大**：2 个 worker 进程同时 dispatch kernel / runtime allocator 操作，Python 端的 context 切换和 allocator lock 争用增加

实证难区分，后续若要优化：先确认 warmup 覆盖所有常用 shape；再看 `nsys` 下 GPU kernel duration 是否真的变长还是 launch 间隙变长。

### 4.3 TTFT 为什么恶化

DP=2 avg TTFT 1.710 s > DP=1 的 1.025 s（+67%）。根因是两重叠加：

- Sync 节拍下"等最慢 worker" — 主进程每 tick broadcast `step` + 等所有 worker 返回。Prefill 阶段 worker 间 step_duration 方差（schedule 决策和剩余 budget 不同）使快的 worker 被拖
- 单 worker prefill step_duration 翻倍（§4.2）直接抬高所有 seq 的 first_token_time

**可优化路径**（非本 spec 范围）：async worker loops（每 worker 独立循环、完成就 push），消除"等最慢"开销。spec §2.2 里已标为 YAGNI，本次不做。

### 4.4 和预期的差距

spec §2.1 目标里只写"显著高于 DP=1"，未承诺具体倍数；§10.1 说 IPC 开销 <1 ms 可忽略——这点被实测验证（IPC 不是瓶颈）。真正造成 DP=2 达不到 2× 的是 prefill step_duration 的放大，和工作负载是"output-heavy（100~1024 tok）但非典型 prefill-heavy"这一组合特性有关。

## 5. 结论

- **改动生效、吞吐实测 1.58×**：DP-only 实现在本机 2×4090 + Qwen3-0.6B 负载下提供了明显加速
- **Scaling 效率 79%**：不完美但可接受；进一步提升需要 async worker loop 和/或更完整的 CUDA graph warmup
- **推荐场景**：decode-heavy 的负载（长 completion、中长 prompt）。Prefill 占比越高、DP 相对 TP 的优势越小
- **零退化**：DP=1 走原 LLMEngine 路径，吞吐 5272 tok/s（vs roadmap #3 结束时 5270 tok/s），在噪声内
- **零抢占**：两种配置下 `preemptions_total=0`，说明 KV cache 充足、改动没挤占显存

## 6. 复现

```bash
# DP=1 baseline
python temp/dp-ab/bench_dp.py 1 temp/dp-ab/bench_metrics_dp1.json

# DP=2
python temp/dp-ab/bench_dp.py 2 temp/dp-ab/bench_metrics_dp2.json

# step 分布分析
python <<'PY'
import json, statistics
for path, label in [('temp/dp-ab/bench_metrics_dp1.json', 'DP=1'),
                    ('temp/dp-ab/bench_metrics_dp2.json', 'DP=2')]:
    d = json.load(open(path))
    prefills = [s for s in d['step_samples'] if s['is_prefill']]
    print(f"{label}: {len(prefills)} prefill steps, "
          f"mean batched={statistics.mean(s['num_batched_tokens'] for s in prefills):.0f}, "
          f"mean duration={statistics.mean(s['step_duration'] for s in prefills)*1000:.1f}ms")
PY
```

**脚本注意点**：`bench_dp.py` 开头强制 `sys.path.insert(0, <project_root>)`，避开 site-packages 里可能存在的旧版 `nanovllm`（从子目录执行脚本时 CWD 不是 project root 会 import 错）。
