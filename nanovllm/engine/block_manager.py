from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 引用计数,表示当前有多少序列在使用该块；当ref_count为0时,表示该块没有被任何序列使用,可以被分配给新的序列；
                            # 当ref_count大于0时,表示该块正在被一个或多个序列使用,不能被分配给新的序列；
                            # 当一个序列完成后,需要将其使用的块的ref_count减1,如果ref_count变为0,则表示该块可以被回收
        
        self.hash = -1  # 该块缓存的token ids的哈希值,用于快速判断一个序列的某块token ids是否与该块缓存的token ids相同；
                        # 当hash为-1时,表示该块没有缓存任何token ids；
                        # 当hash不为-1时,表示该块缓存了一定数量的token ids,并且可以通过hash_to_block_id字典找到对应的块id
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1 # 每次分配时先将ref_count置为1,表示该块正在被一个序列使用；
        self.hash = -1  # 每次分配时先将hash置为-1,表示该块还没有缓存任何token ids；
                        # 当序列append token时,如果该块之前没有缓存满block_size个token ids,那么就继续将新token id添加到该块的token_ids中,并且保持hash为-1；
                        # 如果该块之前已经缓存满block_size个token ids,那么就计算该块缓存的token ids的哈希值,并更新该块的hash和token_ids；
                        # 当序列完成后,回收该块时将hash置为-1,表示该块不再缓存任何token ids
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 需要"取一个出来"([0] 或 remove)、"还一个回去"(append)。deque 两端 O(1)。用 list 的 pop() 是 O(n),慢。
        self.used_block_ids: set[int] = set() # 需要"添加一个进去"(add)、"拿一个出来"(remove)。set 的 add() 和 remove() 都是 O(1)。用 list 的 append() 是 O(1),但 remove() 是 O(n),慢。

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): # prefix cache 的基石
        h = xxhash.xxh64() # 采用级联方式把前一块的 hash 也拌进这一块的 hash 计算里。所以"第 k 块"的 hash 其实是 hash(block_0 ... block_k) 的摘要——它代表了从序列开头到这一块为止的完整内容。
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        ''' 一个冷知识：因为 _deallocate_block 从不清 hash_to_block_id,这张表其实会无限增长,包含很多指向已被别人重新分配、内容已覆盖的"旧 block_id"。
        这些"失效条目"靠 allocate里第二重校验 self.blocks[block_id].token_ids != token_ids 兜住——命中 hash 但内容对不上就当 miss。
        这是实现上的一个简化，生产级 vLLM 会做 LRU 清理。'''
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    #  极简：空闲块数 ≥ 需要的块数。注意这是按最坏情况估算——完全不管 prefix cache 能不能命中。
    # 这是一种保守的策略,避免分配到一半发现没块了、还要回滚。代价是：就算实际上全部命中cache(理论上 0 个新块),也可能因为总数不够而拒绝调度。
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 只有当块满了才计算hash,否则保持hash为-1,表示该块没有缓存任何token ids；这是因为如果块没有满,就算hash命中了也不能复用,因为实际的token ids不完全一样,复用会出问题；只有当块满了,才有资格通过hash命中来复用,因为此时实际的token ids完全一样,复用是安全的
            block_id = self.hash_to_block_id.get(h, -1) 
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # xxhash 有碰撞概率(虽然极低)。如果 hash 相同但 token_ids 实际不同,直接复用会导致 attention 读错内容。这是一道安全保险。 
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else: # 以下两个分支都叫"prefix cache 命中"
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids: # 复用情况
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else: # 不在used池，说明之前_deallocate过了，但hash_to_block_id还没来得及更新，属于"刚好被别人抢先一步复用了"。这种情况不算真正的cache miss，因为块里确实是我们想要的token ids，只不过ref_count需要重新置为1。
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table): # 为什么 reversed？
            '''想想级联 hash 的结构：第 k 块的 hash 依赖第 0..k-1 块。删除一个块不影响 hash 链(hash 只在 allocate/may_append 时写入),所以从语义上顺序或逆序都行。
                但逆序有一个实际优势:后面的块(离序列末端近的)往往是最近才分的独占块(ref_count=1),最可能被真正归还；
                前面的块(prefix 部分)更可能被别人共享(ref_count >1),只是递减。逆序先处理"容易归还的",free 池会更快变大；
                万一这个 deallocate 和别人的 allocate 交替进行,能更早满足对方的需求。这是一个微优化。'''
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1 # 如果当前块里已经有token了,那么就必须要有hash,才能保证append后块里token_ids的正确性；如果当前块里没有token了,那么就不需要hash,保持为-1即可
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
