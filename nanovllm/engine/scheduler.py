from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.metrics import MetricsCollector


class Scheduler:

    def __init__(self, config: Config, metrics: MetricsCollector):
        self.max_num_seqs = config.max_num_seqs # 同时调度的最大序列数，超过这个数的序列需要等待
        self.max_num_batched_tokens = config.max_num_batched_tokens # 每次调度时所有序列总共可以生成的最大token数，超过这个数的序列需要等待
        self.eos = config.eos
        # 该调度器使用块管理器来管理kvcache的分配和回收：
        # 确保每个序列在prefill阶段能够分配到足够的块来缓存其输入token的key和value
        # 在decode阶段能够继续使用这些块来缓存生成token的key和value，同时也能在序列完成后及时回收块以供其他序列使用
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.metrics = metrics
    # 判断是否所有序列都已完成
    def is_finished(self):
        return not self.waiting and not self.running
    # 将新的序列添加到等待队列中，等待下一次调度时被选中进行prefill或decode
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = [] # 本次调度选中的序列列表，最多不超过max_num_seqs；如果是prefill阶段，则这些序列需要生成的token数量之和不超过max_num_batched_tokens；如果是decode阶段，则这些序列每条都只生成1个token
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: # 每次调度时，优先选择等待队列中的序列进行prefill，直到达到最大序列数或最大token数的限制
            seq = self.waiting[0] # 选择等待队列中的第一个序列进行prefill
            
            # 计算该序列本次prefill需要生成的token数量，如果该序列之前已经prefill过了(prefix cache)，那么本次prefill只需要生成剩余的token；
            # 如果该序列之前没有prefill过(prefix cache)，那么至少需要生成1个token来填充kvcache(即使前缀全复用了，也还得把“最后一个 prompt token”喂一次模型，才能拿到“下一个 token”的 logits。)
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1) 
            remaining = self.max_num_batched_tokens - num_batched_tokens # 计算本次调度还剩余的token预算
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):    # no budget
                break
            if remaining < num_tokens and scheduled_seqs:    # 如果剩余的token预算不足以满足该序列的prefill需求，并且已经有其他序列被选中进行prefill了，那么就先不选中该序列，等待下一次调度时再尝试prefill；
                # only allow chunked prefill for the first seq
                break
            if not seq.block_table: # 如果该序列之前没有prefill过，那么先尝试分配块，如果分配失败则跳过该序列，等待下一次调度时再尝试prefill；如果分配成功或者该序列之前已经prefill过了，那么就选中该序列进行prefill
                self.block_manager.allocate(seq)
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = perf_counter()
            seq.num_scheduled_tokens = min(num_tokens, remaining) # 该序列本次prefill需要生成的token数量不能超过剩余的token预算
            if seq.num_scheduled_tokens == num_tokens: # 如果该序列本次prefill能够满足其剩余的token需求，那么就将该序列的状态更新为RUNNING，并从等待队列中移除，加入到正在运行的队列中；如果该序列本次prefill不能满足其剩余的token需求，那么就先不更新该序列的状态，等待下一次调度时继续prefill剩余的token
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq) # 将该序列加入到本次调度选中的序列列表中，并更新本次调度已经选中的序列需要生成的token数量之和
            num_batched_tokens += seq.num_scheduled_tokens
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq) # decode阶段每条序列每次只生成1个token，这样可以更快地响应用户的输入，同时也能更频繁地检查序列是否完成，从而及时回收块资源给其他序列使用
        assert scheduled_seqs   # decode阶段选中的序列都已经从正在运行的队列中移除了，等到它们生成了token并调用postprocess更新状态后，如果没有完成才会被加入到正在运行的队列中
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 将正在运行的序列放回等待队列，并释放其占用的kvcache块，以便其他序列使用
    def preempt(self, seq: Sequence):
        now = perf_counter()
        seq.status = SequenceStatus.WAITING
        seq.preemption_count += 1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.metrics.record_preemption(seq, now)

    # 根据模型输出的token ids更新对应的序列状态，如果是prefill阶段，则更新num_cached_tokens和num_scheduled_tokens；
    # 如果是decode阶段，则将生成的token id添加到completion_token_ids中，并判断是否结束
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
                    seq.num_scheduled_tokens = 0 # 该批次的token还没有真正生成，不更新num_scheduled_tokens，等待下一次调度时继续生成剩余的token
                    continue
            now = perf_counter()
            seq.append_token(token_id)
            seq.token_times.append(now)
            if seq.first_token_time is None:
                seq.first_token_time = now
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = now
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self.metrics.record_finished(seq)
