nano vllm代码调度
整体而言，整个nano vllm的代码调度可以用以下图来概括：

大致流程用户输入prompt，然后其会被运送到LLMEngine.generate 中，然后进行一步步推理，推理完成之后输出，以下是对于此流程的具体代码解析：


example.py
def main():
    # 加载、配置LLM
    path = os.path.expanduser("/path_to_model/qwen3")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 准备prompt
    prompts = [
        "introduce nano vllm",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]

    # 推理，输出回答
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")



LLMEngine.generate
LLMEngine可以看做是整个推理引擎的实例，其内部进行不同请求序列的调度、执行，其中generate方法是其对外暴露的接口，接受prompt、sampling参数，输出模型的回答，简化之后的源码如下（去掉了进度条输出相关逻辑 ）：
def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加新请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
       
        # 处理所有请求直至没有请求可以调度
        while not self.is_finished():
          
            # 执行一步推理：scheduler选择序列 → model推理 → 后处理
            # output: 这一步完成的序列列表 [(seq_id, token_ids), ...]
            output, num_tokens = self.step()
          
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        
        # 处理输出并返回
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
generate的主体包括两个步骤：添加新请求以及调度、执行已有请求，具体为LLMEngine中的add_request、is_finished、step三个方法：
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    # 将prompt转换为Sequence
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)

def step(self):
    # 调度一个待处理请求
    seqs, is_prefill = self.scheduler.schedule()
    # 通过模型处理请求
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens

def is_finished(self):
    return self.scheduler.is_finished()
关于scheduler， 请详见scheduler章节 Scheduler (核心凑批逻辑)


model_runner.run
在调度完成之后，就进入了模型实际推理环节，

def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    #做一些KV cache相关数据结构的准备，大致就是获取一张内存占用列表实现分块管理
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    #准备采样参数供采样器使用
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    #运行模型
    logits = self.run_model(input_ids, positions, is_prefill)
    

#实际运行模型的function
@torch.inference_mode()  # 推理模式，不计算梯度，减少内存占用
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    # 核心条件判断：决定使用哪种推理方式（直接执行 vs CUDA Graph优化执行）
    # 条件1：is_prefill=True 表示当前是预填充阶段（处理输入prompt的初始阶段）
    # 条件2：self.enforce_eager=True 表示强制使用急切执行模式（不使用CUDA Graph优化）
    # 条件3：input_ids.size(0) > 512 表示当前批次大小超过512个序列（执行成本已经大于Cuda Graph的优化）
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # 直接执行模式：逐步进行模型推理
        # 这里是两步操作的组合：
        # 1. self.model(input_ids, positions) - 将输入token和位置信息传入模型主体，得到隐藏状态
        # 2. self.model.compute_logits() - 将隐藏状态通过输出层转换为词汇表logits（未归一化的概率分布）
        #模型完成计算后直接返回logits
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # 使用CUDA Graph优化执行模式（主要用于小批次的decode阶段以提升性能）
        # 。。。（详细见CUDA Graph章节）
        return self.model.compute_logits(graph_vars["outputs"][:bs])

模型计算
关于模型内部结构以及是如何计算的请详见Qwen3
而关于CUDA Graph部分详细见 CUDA Graph 章节
sampler
在模型完成运行以及完成logits的计算之后，返回的logits会被输入到sampler中把logits转换为实际输出token
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    #前略。。。
    
    #准备采样参数供采样器使用
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    #运行模型获得logits
    logits = self.run_model(input_ids, positions, is_prefill)
    
    #把logits转换为实际输出token
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    
    #清理推理上下文状态
    reset_context()
    # 得到token_ids 返回
    return token_ids
sampler内部实现
import torch
from torch import nn

class Sampler(nn.Module):
    # Sampler类：负责从模型输出的logits中采样生成下一个token
    # 支持两种采样策略：贪婪采样(temperature=0)和随机采样(temperature>0)
    # 主要用于控制输出token的随机性

    def __init__(self):
        super().__init__()
        # 采样器不需要可学习的参数，所以__init__方法很简单
        # 主要作用是继承nn.Module，让采样器可以在GPU上运行
        #继承nn.Module后，call这个类相当于直接call 类.forward(参数)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 输入参数：
        # logits: 模型输出的未归一化概率分数，形状 [batch_size, vocab_size]
        # temperatures: 每个序列的温度参数，形状 [batch_size]，控制采样的随机性
        
        # 第一步：将logits转换为float32以提高数值稳定性
        # 例子：如果logits是float16，转换为float32避免精度问题
        logits = logits.to(torch.float)
        
        # 第二步：计算贪婪采样结果（总是选择概率最大的token）
        # 例子：如果词汇表有["hello", "world", "good"]，logits=[2.1, 3.5, 1.8]
        # 则greedy_tokens=1（对应"world"，因为3.5最大）
        greedy_tokens = logits.argmax(dim=-1)  # 形状: [batch_size]
        
        # 第三步：应用温度缩放
        # temperature越大，分布越平滑（更随机）；temperature越小，分布越尖锐（更确定）
        # 例子：原始logits=[2.0, 3.0, 1.0]，temperature=2.0
        # 缩放后：logits=[1.0, 1.5, 0.5]，分布变得更平滑
        logits.div_(temperatures.unsqueeze(dim=1))  # temperatures从[batch_size]扩展到[batch_size, 1]
        
        # 第四步：将logits转换为概率分布
        # softmax将任意实数转换为概率分布（所有值在0-1之间，总和为1）
        # 例子：logits=[1.0, 1.5, 0.5] → probs=[0.24, 0.66, 0.10]
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)  # 形状: [batch_size, vocab_size]
        
        # 第五步：使用Gumbel采样进行随机采样
        # Gumbel采样是一种高效的采样方法，避免了显式的概率采样
        epsilon = 1e-10  # 极小值，防止除零错误
        
        # Gumbel技巧：probs / (exponential_random + epsilon) 的argmax等价于按概率采样
        # exponential_(1)生成指数分布的随机数，这是Gumbel分布的关键组成部分
        # 例子：probs=[0.24, 0.66, 0.10]，随机数=[0.5, 0.2, 1.5]
        # 计算：[0.24/0.5, 0.66/0.2, 0.10/1.5] = [0.48, 3.30, 0.067]
        # argmax选择index=1（对应"world"）
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)
        
        # 第六步：根据温度选择最终结果
        # temperature=0: 使用贪婪采样（总是选择最可能的token）
        # temperature>0: 使用随机采样（按概率分布采样）
        # 例子：如果某个序列的temperature=0，返回greedy_tokens；否则返回sample_tokens
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)  # 形状: [batch_size]

输出实际token
def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        #处理完成获得token_ids
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        #进行后处理，主要检查最后一个token是不是完成符或者太长了，如果不是就继续一直生成直到达到终止条件
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
    
class Scheduler:
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理：处理模型生成的新token
        检查序列是否完成，如果完成就清理资源
        """
        for seq, token_id in zip(seqs, token_ids):
            # 把新生成的token添加到序列中
            seq.append_token(token_id)
            
            # 检查序列是否应该结束：
            # 1. 遇到结束符且没有忽略结束符的设置
            # 2. 达到了最大生成长度限制
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 标记序列为完成状态
                seq.status = SequenceStatus.FINISHED
                # 释放该序列占用的KV cache内存
                self.block_manager.deallocate(seq)
                # 从运行队列中移除
                self.running.remove(seq)
def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        #前略。。。
       
        # 处理所有请求直至所有请求处理完成，输出结果（一整段文字）
        while not self.is_finished():
          
            # 执行一步推理：scheduler选择序列 → model推理 → 后处理
            # output: 这一步完成的序列列表 [(seq_id, token_ids), ...]
            output, num_tokens = self.step()
          
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        
        # 处理输出并返回
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
到此，模型推理完成并输出回复。例子如下图所示：

engine核心机制详解
Scheduler (凑批)

初始化
def __init__(self, config: Config):
    # 一个批次最多能处理多少个序列（比如 8 个请求）
    self.max_num_seqs = config.max_num_seqs
    # 一个批次最多能处理多少个 token（比如 4096 个 token）
    self.max_num_batched_tokens = config.max_num_batched_tokens
    # 结束符的 token ID，用于判断序列是否完成
    self.eos = config.eos
    # KV cache 内存管理器，负责分配和释放显存
    self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
    # 等待队列：新来的请求先放这里排队
    self.waiting: deque[Sequence] = deque()
    # 运行队列：正在生成内容的请求放这里
    self.running: deque[Sequence] = deque()
核心调度逻辑（schedule）
def schedule(self) -> tuple[list[Sequence], bool]:
    """
    凑批：决定这次要一起处理哪些请求
         GPU的并行特性决定了同时处理多个序列比逐个处理效率更高
         通过batching可以充分利用GPU的SIMD（单指令多数据）能力
         最大化硬件资源利用率，降低单次推理的平均延迟
    返回: (要处理的序列列表, 是否为prefill模式)
    
    两阶段策略：
    1. Prefill: 优先处理等待队列中的新请求（首次处理）
    2. Decode: 为运行队列中的请求生成下一个token
    """
    
    # ============ 第一阶段：Prefill（处理新请求）============
    
    # 准备这次要处理的批次
    scheduled_seqs = []  # 最终要一起处理的序列列表
    num_seqs = 0         # 当前批次中的序列数量
    num_batched_tokens = 0  # 当前批次中的总token数量
    
    # 尝试从等待队列中取出新请求组成批次
    #（等待队列代表需要新处理的token）
    # 尝试最大化批次利用率
    while self.waiting and num_seqs < self.max_num_seqs:
        # 取等待队列的第一个请求（先来先服务）
        seq = self.waiting[0]
        
        # 检查两个关键限制条件：
        # 1. token数量限制：加入这个请求后是否超过总token限制
        # 2. 内存限制：是否有足够的显存为这个请求分配KV cache，防止GPU OOM
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
            # 任一条件不满足就停止添加，保持当前批次
            break
        
        # 可以加入批次，执行以下步骤：
        num_seqs += 1  # 批次中序列数量+1
        # 为这个序列分配KV cache内存
        self.block_manager.allocate(seq)
        # 累加token数量（只算新增的，不算已缓存的）
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        # 标记序列状态为"运行中"
        seq.status = SequenceStatus.RUNNING
        # 从等待队列移除，加入运行队列，同时加入这次要处理的批次
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    
    # 如果在之前的操作中发现有新序列要处理，判断处于prefill模式（因为Decode是生成新token）
    if scheduled_seqs:
        return scheduled_seqs, True  # 返回，True表示prefill模式

    # ============ 第二阶段：Decode（生成新token）============
    # 如果没有新请求要处理，判断处于decode模式，开始根据running中的token生成下一个token
    while self.running and num_seqs < self.max_num_seqs:
        # 从运行队列取出一个序列
        seq = self.running.popleft()
        
        # ============ 抢占机制：解决动态内存需求问题 ============
        # 为什么需要抢占？
        # 1. 大语言模型推理时序列长度动态增长，无法预测最终内存需求
        #    例子：用户问"写一首诗"，系统预期200token，但用户追问"再写10首"
        #    导致序列从200token增长到2000+token，内存需求增长10倍
        # 2. 不同序列增长速度不同，有些可能远超预期长度
        #    例子：A序列"简单问答"很快完成，B序列"长篇小说"持续增长
        #    B序列可能占用越来越多内存而A序列已经释放，导致内存分配不均
        # 3. 如果不抢占，内存不足时整个系统会卡死
        #    例子：4个序列各占用50%内存，当其中一个需要更多内存时
        #    系统无法分配，所有序列都卡死等待，整个服务不可用
        #
        while not self.block_manager.can_append(seq):
            # 内存不够时的抢占策略：踢掉其他序列来释放内存
            if self.running:
                # 踢掉运行队列中的最后一个序列（后进先出，LIFO）
                # 这样避免了随机抢占，形成稳定的轮转机制
                # 随机抢占的问题：可能反复踢掉同一序列导致饥饿，系统行为不可预测，难以调试优化
                self.preempt(self.running.pop())
            else:
                # 如果运行队列空了，就踢掉当前序列
                # 这是最后的保险措施，确保系统不会死锁
                self.preempt(seq)
                break
        else:
            # 内存够用，可以继续处理这个序列
            num_seqs += 1  # 批次序列数量+1
            # 允许序列使用更多内存（为新token分配空间）
            self.block_manager.may_append(seq)
            # 加入这次要处理的批次
            scheduled_seqs.append(seq)
    
    # decode阶段必须有序列要处理
    assert scheduled_seqs
    # 把处理过的序列重新放回运行队列的开头（reversed保持原顺序）
    self.running.extendleft(reversed(scheduled_seqs))
    # 返回decode模式
    return scheduled_seqs, False  # False表示decode模式

其他方法
def is_finished(self):
    """检查是否所有任务都完成了"""
    # 当两个队列都为空时，说明没有任务需要处理
    return not self.waiting and not self.running

def add(self, seq: Sequence):
    """添加新的请求到等待队列"""
    # 新请求先到等待队列末尾排队，遵循先来先服务原则
    self.waiting.append(seq)

def preempt(self, seq: Sequence):
    """
    抢占（踢出）一个序列：当内存不够时使用
    被踢出的序列会重新排队，不会丢失
    
    抢占机制的公平性保证：
    1. 被抢占的序列不会丢失，而是重新排队
    2. 使用appendleft放到等待队列开头，下轮优先处理
    3. 避免了序列饥饿问题，每个序列最终都会完成
    4. 抢占顺序固定（LIFO），行为可预测，不会出现混乱
    """
    # 改变序列状态为"等待"
    seq.status = SequenceStatus.WAITING
    # 释放该序列占用的KV cache内存
    self.block_manager.deallocate(seq)
    # 重新放到等待队列的开头（优先处理，因为之前已经开始过）
    # 这确保了被抢占的序列能够快速重新获得处理机会
    self.waiting.appendleft(seq)

def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    """
    后处理：处理模型生成的新token
    检查序列是否完成，如果完成就清理资源
    """
    for seq, token_id in zip(seqs, token_ids):
        # 把新生成的token添加到序列中
        seq.append_token(token_id)
        
        # 检查序列是否应该结束：
        # 1. 遇到结束符且没有忽略结束符的设置
        # 2. 达到了最大生成长度限制
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            # 标记序列为完成状态
            seq.status = SequenceStatus.FINISHED
            # 释放该序列占用的KV cache内存
            self.block_manager.deallocate(seq)
            # 从运行队列中移除
            self.running.remove(seq)
KV Cache


KVCache概念总览
1，KV cache 是什么
1. 在 Transformer 解码器（GPT 类模型）中，每一层都会把输入向量拆分成三组矩阵：
• Q（Query）
• K（Key）
• V（Value）
2. 在自回归生成时，第 t 步要让当前 Query 与 0…t-1 步对应的 K、V 做注意力计算。
3. 在Decoder模型中，第 0…t-1 步的 K、V 在之前已经算过，而且它们不再改变。
4. “KV cache” 就是把这些已经算好的 K、V 按层保存下来，后续步骤直接拿来用，而不是再次前向传播去重新计算。
2，为什么可以加速
无缓存：
• 第 t 步需要把长度为 t 的整个序列重新送进模型的所有层，重新得到 0…t 的 K、V。
• 计算量与序列长度 t 成正比。
有缓存：
• 第 t 步只把“新来的那个 token”过一遍模型，得到它自己的 K_new、V_new。
• 把 K_new、V_new 追加到缓存里；之前 0…t-1 的 K、V 直接复用。
• 计算量与 1 成正比（只处理 1 个 token），与 t 无关。
因此当序列变长时，加速比≈t : 1。
3，假如我们生成 4 个 token。假设每次前向传播 1 个 token 的耗时为 1 单位。
1. 不使用 KV cache
步骤1：序列长度 1，耗时 1
步骤2：序列长度 2，耗时 2
步骤3：序列长度 3，耗时 3
步骤4：序列长度 4，耗时 4
总耗时 1+2+3+4 = 10
2. 使用 KV cache
每一步只处理新 token，耗时 1
总耗时 1+1+1+1 = 4
加速倍率居然达到了 = 10 / 4 = 2.5。
PagedAttention
但是，传统的KVcache弊端也比较明显，由于是典型的空间换时间优化，那么当上下文越长，cache矩阵占用的内存也会越多，而且传统的KVcache为每个序列分配连续的KV缓存，导致内存碎片。
# 传统方式的问题
seq1: [████████████        ]  # 浪费空间
seq2: [████████            ]  # 浪费空间
seq3: [████████████████    ]  # 浪费空间
而PagedAttention的解决方案是将KV缓存分割成固定大小的块（类似OS的分页机制）。
Block Pool: [████][████][████][████][████][████]
seq1: 使用 Block[0,1,2]
seq2: 使用 Block[3,4]
seq3: 使用 Block[5] + 新分配的块
具体原理如下图

通过BlockTable，连续的逻辑Kv Blocks被映射为不连续的物理的Kv Blocks，解决了内存碎片的问题。

Prefix Caching
在PagedAttention中，KV Cache只是在一个请求内复用，而没有做到跨请求的KV Cache复用。在多轮对话的场景下，下一轮的prompt其实刚好就是上一轮的prompt+completion：

如果新一轮的prompt的KV Cache能够直接复用上一轮计算好的结果，做到跨请求复用KV Cache，那么就可以显著提升prefill的性能，降低新一轮请求的TTFT（Time To First Token）。这种优化方法被称为Prefix Caching，核心思想是缓存系统提示和历史对话中的键值（KV）缓存，以便在后续请求中重用，从而减少首次Token的计算耗时。在vllm中，哈希码作为物理KV Block的唯一标识，这样内容一致的块会直接拿来复用，大致流程如下：
block = 4 tokens
① seqA arrives
   101 102 103 104 | 105 106 107 108
   └────── P0 ─────┘ └────── P1 ─────┘   (ref=1  ref=1)

   BlockPool : [P0] [P1]

② seqB arrives
   101 102 103 104 | 105 106 107 108 | 120 121 122 123
   └──────────┬────┘ └──────────┬────┘ └────── P2 ─────┘
              ↓                 ↓
            P0 (ref=2)       P1 (ref=2)

   BlockPool : [P0] [P1] [P2]

③ seqC arrives
   200 201 202 203
   └────── P3 ─────┘   (ref=1)

   BlockPool : [P0] [P1] [P2] [P3]

④ seqA ends → P0、P1 ref-- → 1      (仍被 seqB 占用)

最终显存里只存一份 P0/P1，所有带相同前缀的请求都指向它们。
总结：相同 token-block → 哈希命中共享；ref=0 时块立即回收。

nano-vllm实现
首先，KVcache会在ModelRunner中进行初始化
class ModelRunner:
    def __init__():
        # 保存配置和分布式训练相关参数以及初始化分布式训练环境
        #。。。
        
        # 初始化的关键步骤
        self.warmup_model()                               # 预热模型：运行一次推理来清理显存和优化
        self.allocate_kv_cache()                         # 分配KV缓存：为注意力机制预分配显存
        if not self.enforce_eager:                       # 如果不强制急切执行
            self.capture_cudagraph()                     # 捕获CUDA Graph：预先记录计算图以优化性能


    def allocate_kv_cache(self):
        # KV缓存分配：为注意力机制预分配Key和Value的缓存空间
        
        free, total = torch.cuda.mem_get_info()  # free: 可用显存, total: 总显存
        #获取一张卡上 global free and total GPU memory.
        used = total - free  #当前全部已被使用的显存
        
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]      # 历史峰值显存使用
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前显存使用
        #PyTorch 自己的内存池的使用情况
        
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        
        # 计算单个缓存块的字节大小，2 分别代表Key和Value
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        
        # 可用于KV blocks的显存 除以单块字节数得到最大可分配的块数 
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        
        assert config.num_kvcache_blocks > 0
        
        #分配统一的KV缓存张量，形状：[2, 层数, 块数, 块大小, KV头数, 头维度]
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        
        # 遍历模型的所有模块，找到有KV缓存属性的注意力层，将缓存绑定到各个注意力层
        layer_id = 0
        for module in self.model.modules():
            # 检查模块是否有k_cache和v_cache属性（注意力层的标志）
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 将统一缓存的对应部分分配给该层
                module.k_cache = self.kv_cache[0, layer_id]  # Key缓存：第0维
                module.v_cache = self.kv_cache[1, layer_id]  # Value缓存：第1维
                layer_id += 1  # 移动到下一层

目前来说，我们创建和记录了总的KVCache池，根据之前的设计思路我们了解到，除了一大块KVcache之外，我们还需要还需要准备一个BlockTable来实现分块逻辑，所以我们就需要BlockTable，这是其在vllm中的实现：
Block
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

class BlockManager:
    """
    KV缓存块管理器
    
    负责管理所有的KV缓存块，提供以下核心功能：
    1. 内存分配：为新序列分配缓存块
    2. 内存释放：回收不再使用的缓存块  
    3. Prefix Cache：通过哈希快速找到相同前缀的缓存
    4. 引用管理：跟踪每个块的使用情况

    """

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size                                    # 每个块的大小（256个token）
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 创建所有的缓存块
        self.hash_to_block_id: dict[int, int] = dict()                 # 哈希值 → 块ID的映射表，用于prefix cache查找
        self.free_block_ids: deque[int] = deque(range(num_blocks))     # 空闲块ID队列，先进先出
        self.used_block_ids: set[int] = set()                          # 正在使用的块ID集合

#省略一部分基础function

    def allocate(self, seq: Sequence):
        """
        为序列分配KV缓存块
        
        处理流程：
        1. 将序列按block_size分成多个块
        2. 对每个块计算哈希值（包含前缀信息）
        3. 查找是否已有相同内容的缓存块
        4. 缓存命中：复用现有块，增加引用计数
        5. 缓存未命中：分配新块
        
        """
        assert not seq.block_table  # 确保序列还没有分配过块
        h = -1                      # 前缀哈希值
        cache_miss = False 
        
        # 对于seq需要的每个块都遍历一遍，分开allocate
        for i in range(seq.num_blocks):
            
            token_ids = seq.block(i)
            
            # 计算当前块的哈希值以用于 prefix caching, 只有满块（256个token）才计算哈希，未满块设为-1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # 查找哈希表，看是否已经有相同内容的块
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 检查是否真的缓存命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 缓存未命中，需要分配新块
                
            if cache_miss:
                # === 缓存未命中：分配新的块 ===
                block_id = self.free_block_ids[0]  # 取第一个空闲块
                block = self._allocate_block(block_id)
            else:
                # === 缓存命中：复用现有块 ===
                seq.num_cached_tokens += self.block_size  # 增加已缓存token数量
                if block_id in self.used_block_ids:
                    # 块正在被其他序列使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块是空闲的，重新激活
                    block = self._allocate_block(block_id)
            
            # 如果计算了哈希值，更新块的信息和索引
            if h != -1:
                block.update(h, token_ids)        # 更新块的哈希和内容
                self.hash_to_block_id[h] = block_id  # 建立哈希→块ID的映射
                
            # 将块ID添加到序列的块表中
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的所有缓存块
        
        当一个对话结束时，需要释放它占用的所有缓存块：
        1. 遍历序列的所有块
        2. 减少每个块的引用计数
        3. 引用计数为0时，释放块到空闲队列
            
        注意：使用引用计数机制，多个序列可以安全共享同一个缓存块
        """
        # 反向遍历块表（从后往前释放）
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数
            
            # 如果没有序列在使用这个块了，释放它
            if block.ref_count == 0:
                self._deallocate_block(block_id)
                
        # 清空序列的缓存信息
        seq.num_cached_tokens = 0
        seq.block_table.clear()


    def may_append(self, seq: Sequence):
        """
        处理序列追加token时的块管理
        
        当模型生成新token时，需要动态管理缓存块：
        1. 如果开始填充新块：分配一个新的空闲块
        2. 如果刚好填满一个块：计算哈希值并建立索引
        3. 如果在块中间：不需要特殊处理
            
        举例说明：
        - 假设block_size=256，当前序列有511个token
        - 512 % 256 = 0，说明刚好填满第2个块
        - 需要计算这个满块的哈希值，用于后续的prefix cache查找
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 获取当前序列的最后一个块
        
        if len(seq) % self.block_size == 1:
            # === 情况1：刚开始填充新块 ===
            # 序列长度模block_size等于1，说明：
            # - 之前的块都已经满了（256, 512, 768...）
            # - 现在开始填充新的第N+1个块
            # - 需要分配一个新的空闲块来存储接下来的token
            assert last_block.hash != -1  # 上一个块应该已经有哈希值
            block_id = self.free_block_ids[0]      # 取第一个空闲块
            self._allocate_block(block_id)         # 分配这个块
            block_table.append(block_id)           # 添加到序列的块表中
            
        elif len(seq) % self.block_size == 0:
            # === 情况2：刚好填满一个块 ===
            # 序列长度是block_size的整数倍，说明：
            # - 当前块刚好被填满（256, 512, 768...个token）
            # - 需要计算这个满块的哈希值
            # - 建立哈希索引，用于后续的prefix cache查找
            assert last_block.hash == -1          # 当前块应该还没有哈希值
            token_ids = seq.block(seq.num_blocks-1)  # 获取最后一个块的token序列
            
            # 获取前一个块的哈希值作为前缀
            # 如果只有一个块，前缀为-1
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            
            # 计算当前块的哈希值（包含前缀信息）
            h = self.compute_hash(token_ids, prefix)
            
            # 更新块的信息并建立索引
            last_block.update(h, token_ids)       # 更新块的哈希和token内容
            self.hash_to_block_id[h] = last_block.block_id  # 建立哈希→块ID的映射
            
        else:
            # === 情况3：在块中间 ===
            # 序列长度不是特殊值，说明：
            # - 当前块还在填充中（比如有200个token，还能再加56个）
            # - 不需要分配新块，也不需要计算哈希
            # - 什么都不做，等待继续添加token
            assert last_block.hash == -1          # 未满的块不应该有哈希值
以及这是缓存块的实现
class Block:
    """
    KV缓存块
    
    每个Block存储固定数量token的KV缓存信息，包括：
    - 块的唯一标识
    - 引用计数（有多少个序列在使用这个块）
    - 内容哈希值（用于快速查找相同内容）
    - 存储的token序列
    """

    def __init__(self, block_id):
        """
        初始化一个空的缓存块
        
        block_id: 块的唯一标识符，就像格子编号
        """
        self.block_id = block_id        # 块的唯一编号，用于标识这个缓存块
        self.ref_count = 0              # 引用计数：有多少个序列正在使用这个块
        self.hash = -1                  # 内容哈希值：用于快速识别是否是相同内容，-1表示未设置
        self.token_ids = []             # 存储的token序列：这个块实际缓存的token内容

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的内容信息
        
        当块被填满256个token时，计算并存储这些token的哈希值，
        用于后续的prefix cache查找
        
        Args:
            hash: 计算出的内容哈希值，用于快速匹配
            token_ids: 这个块存储的完整token序列
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块为初始使用状态
        
        当一个空闲块被重新分配时调用：
        - 设置引用计数为1（有一个序列开始使用）
        - 清空哈希值和token内容
        """
        self.ref_count = 1              # 新分配的块初始引用计数为1
        self.hash = -1                  # 重置哈希值，表示内容未确定
        self.token_ids = []             # 清空之前的token内容
关于Block是何时被写入的（may_append 是什么时候被调用的），可见Scheduler (核心凑批逻辑)相关代码。
Slot mapping
看完Block的创建和分配逻辑之后，基于Block ID还有更精细的一层从逻辑到现实内存地址的mapping，被称为slot_mapping，大致作用便是给存储KV缓存的新token做从逻辑层到物理层的mapping，便于多线程的写入和更为精细的内存管理，在Attention Layer中其主要代替Block table做从逻辑层到物理层的映射，其计算方式在model_runner.py 的 prepare_prefill 和 prepare_decode 方法中，具体代码可见：
prepare_prefill:
   def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []      # 存放所有要处理的token
        positions = []      # 存放每个token的位置信息 
        cu_seqlens_q = [0]  # Query序列的累积长度
        cu_seqlens_k = [0]  # Key序列的累积长度
        max_seqlen_q = 0    # 最大Query长度 - 用于内存预分配
        max_seqlen_k = 0    # 最大Key长度 - 用于内存预分配
        slot_mapping = []   # KV缓存的内存位置映射
        block_tables = None
        
        # 遍历每个序列，收集需要处理的token
        for seq in seqs:
            seqlen = len(seq) 
            
            # 只处理尚未缓存的token部分(避免重复计算)。例如: 序列长度100，已缓存80个token，只需处理后20个
            input_ids.extend(seq[seq.num_cached_tokens:])
            
            # 添加对应的位置编码(告诉模型每个token在序列中的位置)。例如: 已缓存80个，新的20个token位置为[80,81,82,...,99]
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens  # Query长度 = 需要新计算的token数
            seqlen_k = seqlen                          # Key长度 = 序列总长度(包括缓存)
            
            
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)#记录新token边界，
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)#记录所有token边界
            # 例如：2个序列[总长6已缓存4, 总长5已缓存3] → cu_seqlens_q=[0,2,4], cu_seqlens_k=[0,6,11]
            
            # 记录最大长度 - GPU可以根据最大长度预分配计算资源，避免动态分配开销
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:
                continue
                
            # 这个循环的逻辑：
            # 1. 遍历未缓存的块：只处理需要存储新KV的块
            # 2. 逐块展开位置：将每个块内的位置展开成连续数字
            # 3. 构建占用地图：告诉GPU这些具体位置需要存储新KV
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                # 计算物理块的起始位置
                # 例如：block_table[i]=5, block_size=16 -> start=80
                start = seq.block_table[i] * self.block_size
                
                # 计算块的结束位置
                if i != seq.num_blocks - 1:  # 完整块
                    end = start + self.block_size  # 例如：[80,81,...,95] 共16个位置
                else:  # 最后一个块 - 避免为空位置浪费计算
                    # last_block_num_tokens会随序列生成动态增长
                    # 例如：序列50个token时只用2个位置[80,81]，生成到52个token时用4个位置[80,81,82,83]
                    end = start + seq.last_block_num_tokens 
                
                # 展开块内所有位置，添加到"占用地图"
                # 例如slot_mapping = [80,81,...,95, 112,113,114]
                slot_mapping.extend(list(range(start, end)))
                
        # 前缀缓存检测：如果总token数 > 新token数，说明有历史KV缓存
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache enabled
            block_tables = self.prepare_block_tables(seqs)#将每个序列的block_table填充到相同长度，方便GPU并行处理
            
        # 转换为GPU张量 - 使用pin_memory和non_blocking加速CPU到GPU传输
        input_ids = torch.tensor(input_ids, dtype=torch., pin_memory=True).cuda(non_blocking=True)  # 输入token ID序列，int64确保支持大词汇表
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)  # token位置编码，int64支持长序列位置索引
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # 查询序列累积长度，int32足够表示序列边界
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # 键序列累积长度，用于注意力计算的序列分割
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # KV缓存槽位映射，指向具体的缓存位置
        
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                   slot_mapping, None, block_tables)
        return input_ids, positions
prepare_decode:
def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []     # 存放每个序列的最后一个token(即将预测下一个的token)
        positions = []     
        slot_mapping = []  
        context_lens = []  # 每个序列的上下文长度(用于注意力计算)
        
        # 遍历每个序列，收集decode所需的信息
        for seq in seqs:
            # 只取最后一个token - decode每次只处理一个新token
            input_ids.append(seq.last_token)
            
            # 下一个token的位置 = 当前序列末尾
            positions.append(len(seq))
            
            # 记录上下文长度 - 告诉注意力机制要关注多少历史token
            context_lens.append(len(seq))
            
            # 计算新token在KV缓存中的存储位置
            # 新token存储在最后一个块的下一个可用位置
            # 例如: 最后块号=3，块大小=16，已用=12 -> 位置 = 3*16+12-1 = 59
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        # 转换为GPU张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # Decode模式总是需要block_tables来访问历史KV缓存
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置推理上下文 - False告诉模型当前是decode模式
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions


Attention Layer
到目前为止，我们了解了Block到Slotmapping的转换，现在，我们可以看一下其在Attention Layer是具体如何被使用的，首先这是Attention Layer和KVCache相关的代码：

class Attention(nn.Module):
    def __init__():
        
        # 初始化为空张量，但实际会被之前Model Runner中缓存绑定的逻辑替代
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor 
        q = q.view(-1, self.num_heads, self.head_dim)      
        k = k.view(-1, self.num_kv_heads, self.head_dim)  
        v = v.view(-1, self.num_kv_heads, self.head_dim) 
        context = get_context()
        
        # 获取KV缓存的引用
        k_cache, v_cache = self.k_cache, self.v_cache
        
        #增量存储新的K、V到缓存
        if k_cache.numel() and v_cache.numel():  # numel()>0表示缓存已初始化
            # GPU并行存储
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:  # Prefill模式：处理新的请求
            
            if context.block_tables is not None:    # 如果有cache
                k, v = k_cache, v_cache  # 直接取出缓存的KV
            
            # 使用Flash Attention处理序列的并行计算
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q,    # Q最长序列长度，用于GPU内存分配
                                       cu_seqlens_q=context.cu_seqlens_q,    # Q序列边界[0,5,12,20]，告诉GPU哪段属于哪个序列
                                       max_seqlen_k=context.max_seqlen_k,    # K最长序列长度，同Q
                                       cu_seqlens_k=context.cu_seqlens_k,    # K序列边界，支持批量并行计算
                                       softmax_scale=self.scale,             
                                       causal=True,                         
                                       block_table=context.block_tables)     # KV缓存内存块位置表
        else: # Decode模式
            o = flash_attn_with_kvcache(q.unsqueeze(1),           # 在seq维度增加维度：[tokens,1,heads,dim]
                                        k_cache, v_cache,         # 直接使用缓存的历史K、V，因为肯定有
                                        cache_seqlens=context.context_lens,  # 每个序列已缓存的长度
                                        block_table=context.block_tables,    # GPU内存块映射表
                                        softmax_scale=self.scale,            
                                        causal=True)                        
        
        # 从多头格式 [tokens, heads, head_dim] 变回平铺格式 [tokens, heads * head_dim]
        # 确保输出格式与输入格式一致
        o = o.view(-1, self.num_heads * self.head_dim)
        
        return o


CUDA Graph
CUDA Graph是CUDA 10.0引入的一项技术，它将一系列CUDA操作"记录"成一个静态图，然后通过call CUDA Graph replay可以高效地重复执行这个图。传统的CUDA kernel启动有较高的CPU开销，而CUDA Graph通过批量提交和优化执行路径显著降低了这种开销。
传统方式: CPU逐个提交kernel → GPU执行 → 反复多次
Graph方式: CPU录制图 → GPU批量执行整个图 → 高效重复
nano-vllm中的CUDA Graph实现
# nanovllm/engine/model_runner.py
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # ... 基础初始化 ...
        if not self.enforce_eager:
            self.capture_cudagraph()  # 关键：初始化时capture图

    @torch.inference_mode()
    def capture_cudagraph(self):
        """CUDA Graph捕获主流程"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 准备静态tensor缓冲区
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 预定义支持的batch sizes
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 为每个支持的batch size分别捕获图
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], 
                       context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # Warmup: 先执行一次让GPU分配好内存
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 关键：实际图捕获
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
                
            if self.graph_pool is None:
                self.graph_pool = graph.pool()  # 创建内存池复用
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
    
        # 保存静态变量引用供replay使用
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
大致调用流程为：
# nanovllm/engine/model_runner.py
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    # 判断是否使用CUDA Graph
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # 直接执行，不使用图
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # 使用CUDA Graph执行
        bs = input_ids.size(0)
        context = get_context()
        
        # 选择合适的图：找到>=当前batch size的最小图
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        
        # 重置graph中的输入变量（除了outputs）为零，准备接收新的输入数据
        for k, v in graph_vars.items():
            if k != "outputs": #output会被重写，所以不需要初始化
                v.zero_()
        
        # 关键：将当前数据复制到静态缓冲区
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        
        # 执行图：一次性运行所有kernel
        graph.replay()
        
        # 提取结果
        return self.model.compute_logits(graph_vars["outputs"][:bs])
完整流程大概是这样：
# CUDA Graph在nano-vllm中的完整流程

初始化阶段:
模型加载 → 权重分配 → KV Cache分配 → CUDA Graph捕获
     ↓
为每个batch size(1,2,4,8,16,32...)分别捕获静态图
     ↓
保存图引用和静态tensor缓冲区

推理阶段:
新请求到达 → 判断是否使用Graph
     ↓
是decode且batch<=512 → 选择合适的Graph → 数据复制到静态缓冲区 → replay() → 提取结果
     ↓
否则 → 直接执行模型forward

性能优势:
- CPU开销: 从O(num_layers)降低到O(1)
- GPU执行: 批量提交，优化内存访问模式
- 适用场景: decode阶段的重复计算模式
- 限制条件: 静态shape，小batch效果最佳

Qwen3
当model_runner进行到self.model这一步时，代码会进入到Qwen3的模型内部开始实际的推理计算，这点从代码中也可以看出
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 保存配置和分布式训练相关参数
        self.config = config                    # 模型运行配置
        hf_config = config.hf_config           # HuggingFace模型配置

        #。。。略
    
        # 初始化模型
        self.model = Qwen3ForCausalLM(hf_config)          # 创建Qwen3语言模型
        load_model(self.model, config.model)              # 加载预训练权重
        
@torch.inference_mode() 
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # 直接执行模式。。。略
        
        #input_ids: 输入文本转换成的token ID序列，prefill阶段包含完整prompt的所有token，decode阶段只包含单个新生成的token
        #positions: 每个token在完整序列中的绝对位置索引（从0开始），用于RoPE位置编码让模型理解token的顺序关系
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # 使用CUDA Graph。。。（略
        
        #graph_vars["outputs"][:bs] CUDA Graph缓存的模型replay出来的结果 （相当于模型输出）
        #[:bs]是相当于取当前批次输入对应的部分
        return self.model.compute_logits(graph_vars["outputs"][:bs])
Qwen3ForCausalLM
进入到模型内部之后，首先能看到的是Qwen3ForCausalLM
class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3ForCausalLM：完整的文本生成模型
    
    架构组成:
    - Qwen3Model: 主干网络 (token嵌入 + 28层解码器 + 最终归一化)
    - ParallelLMHead: 语言建模头 (1024维 → 151,936词汇表概率分布)
    - 权重绑定: embed_tokens.weight = lm_head.weight (tie_word_embeddings=true)
    """
    
    # 打包模块映射：用于权重加载时的参数名称转换
    # 将HuggingFace格式的参数名映射到本实现的参数名
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # query投影 → 合并的QKV投影中的Q部分
        "k_proj": ("qkv_proj", "k"),      # key投影 → 合并的QKV投影中的K部分  
        "v_proj": ("qkv_proj", "v"),      # value投影 → 合并的QKV投影中的V部分
        "gate_proj": ("gate_up_proj", 0), # 门控投影 → 合并的门控上升投影中的第0部分
        "up_proj": ("gate_up_proj", 1),   # 上升投影 → 合并的门控上升投影中的第1部分
    }

    def __init__(
        self,
        config: Qwen3Config  # 真实配置：151936词汇，1024维，28层，16头注意力
    ) -> None:
        super().__init__()
        # 主干模型：处理序列的核心网络
        self.model = Qwen3Model(config)
        
        # 语言建模头：将1024维隐藏状态映射到151936维词汇表概率
        # 支持张量并行，在多GPU环境下分布式计算
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 权重绑定：embed_tokens和lm_head共享权重矩阵 (tie_word_embeddings=true)
        # 这是一种常见的优化技术，减少参数量同时保持性能
        if config.tie_word_embeddings:
            # 将嵌入层权重与语言建模头权重绑定为同一个tensor
            # 151936×1024的权重矩阵被两个模块共享使用
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,    # 输入token序列 [batch_size, seq_len]
        positions: torch.Tensor,   # token位置索引 [batch_size, seq_len]
    ) -> torch.Tensor:
        # 通过主干模型处理输入序列
        # 经过词嵌入 → 28层解码器 → 最终归一化，得到1024维隐藏状态
        hidden_states = self.model(input_ids, positions)  # [batch_size, seq_len, 1024]
        
        # 返回隐藏状态，供语言建模头计算词汇表概率分布
        # 注意：这里只返回隐藏状态，不直接计算logits
        # logits计算由compute_logits方法单独处理
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # 模型输出的隐藏状态 [batch_size, seq_len, 1024]
    ) -> torch.Tensor:
        #。。。先略
Qwen3ForCausalLM 类主要是简单的对一些参数的初始化和调用，因为初始化成了nn.Module，call这个类相当于直接call 类.forward, 所以输入会被导入到Qwen3Model()也就是实际的模型网络

Qwen3Model
到这一部分开始，输入进入到了实际网络，nano-vllm里面的实现是dense版本（非MOE）所以整个Qwen3的网络结构如下所示：

对应的Qwen3Model代码：
class Qwen3Model(nn.Module):
    """
    Qwen3模型主体：从token输入到最终隐藏状态输出的完整处理流程
    """

    def __init__(
        self,
        config: Qwen3Config,  # 包含所有模型超参数的配置对象
    ) -> None:
        super().__init__()
       
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,    # 输入token IDs [batch_size, seq_len]
        positions: torch.Tensor,   # token位置索引 [batch_size, seq_len]  
    ) -> torch.Tensor:
        
        hidden_states = self.embed_tokens(input_ids)  # [batch_size, seq_len, 1024]
        
        residual = None  # 残差连接初始化
        
        for layer in self.layers: 
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)  # [batch_size, seq_len, 1024]
        return hidden_states
而本文会一步步拆解此网络结构模型并展示相对应的实际计算代码，首先从第一步Embedding开始：
Embedding

class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,  # 包含所有模型超参数的配置对象
    ) -> None:
        super().__init__()
        # 参数：
        #vocab_size: 151,936个token（相当于15万个单词的字典）
        #hidden_size: 1,024维（每个单词用1,024个数字来表示）
        # 使用张量并行优化，支持多GPU分布式训练
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,    # 输入token IDs [batch_size, seq_len]
        positions: torch.Tensor,   # token位置索引 [batch_size, seq_len]  
    ) -> torch.Tensor:
        # 第一步：词嵌入 - 将token IDs转换为1024维向量表示
        # 例子：input_ids=[101, 8092, 2088] → 每个ID转为1024维向量
        hidden_states = self.embed_tokens(input_ids)  # [batch_size, seq_len, 1024]
        
        #。。。略
而在VocabParallelEmbedding内部：

class VocabParallelEmbedding(nn.Module):
    """
    并行词汇嵌入层：多GPU协作处理超大词汇表的完整分布式系统
    """

    def __init__(
        self,
        num_embeddings: int,    # 词汇表总大小，151,936个token
        embedding_dim: int,     # 每个词的向量维度，1,024维
    ):
        super().__init__()
        
        #。。。忽略TP并行逻辑，这一段只展示计算操作
        #大致就是把输入拆成几块分散到不同GPU上（之后在TP并行章节详细讲解）
        self.num_embeddings = num_embeddings  # 总共15万个词
        
        # 创建嵌入权重矩阵
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor):
        
        # 在词汇表里查找词向量
        # 只查这块GPU负责的词，其他词返回零向量或占位向量
        # 把token ID转换成稠密的词向量表示
        # ==================== F.embedding(x, self.weight) 操作解释 ====================
        # 输入 x: token ID序列，比如 [2, 3, 5] (对应"我"、"爱"、"苹果")
        # 权重 self.weight: 词汇嵌入矩阵，形状 [词汇表大小, 向量维度]，比如 [50000, 768]
        #
        # 操作过程：
        # token_id=2 → 取嵌入矩阵第2行 → 得到768维向量 [0.1, 0.3, -0.2, ...]
        # token_id=3 → 取嵌入矩阵第3行 → 得到768维向量 [0.5, -0.1, 0.8, ...]  
        # token_id=5 → 取嵌入矩阵第5行 → 得到768维向量 [0.2, 0.7, -0.5, ...]
        #
        # 输出 y: 词向量矩阵，形状 [3, 768]，每行是一个词的向量表示
        # 本质：把离散的token ID转换成连续的稠密向量，供神经网络处理
        y = F.embedding(x, self.weight)
        
        return y

RMSNorm

在单层实现中，RMSNorm进行了4次，分别在进入Attention Layer之前，进入Attention Layer之初还有进行MLP Layer之前和进行MLP之后，主要作用便是把数据"压缩"到稳定范围，防止输出变得极端（以及和训练时结构一致），具体代码体现在：
class Qwen3Attention(nn.Module):

    def __init__() -> None:
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head) #Q RMSNorm
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape) # K RMSNorm
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output

class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention()
        self.mlp = Qwen3MLP()
        # config.rms_norm_eps = 1e-06 - RMS归一化eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #residual：负责保留输入完整信息，这样第一层的输入传到最后不会变形的太严重
        #第一次没有，所以直接RMSNorm，后续的话使用融合residual操作的RMSNorm，减少kernel调用，提升效率
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states) #第一次RMSNorm
        else:
            #进入Qwen3DecoderLayer后，每一层在计算输入前给Q，K都要进行一次RMSNorm+residual连接操作
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        #在attention layer中每次计算也要进行对于Q和K的RMSNorm操作
        # * 不对V进行是因为V不参与相似度计算，V 只在最后被加权，不影响注意力权重分布
        hidden_states = self.self_attn(positions, hidden_states)
        # 在离开attention层后，第二次残差连接和层归一化
        # 将attention的输出与residual融合，稳定数据为MLP前馈网络做预处理
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        #最终层归一化：对模型输出进行归一化
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
具体计算实现：
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps # 数值稳定性参数，防止除零错误
        # 可学习的权重参数，初始化为全1，让模型能调整归一化的强度
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        标准的 RMS 归一化：把数据"压缩"到稳定范围
        
        计算例子：输入 [3.0, 6.0, 9.0]
        1，平方: [9, 36, 81]
        2，平均: (9+36+81)/3 = 42
        3，开方: √42 ≈ 6.48 (这就是RMS)
        4，归一化: [3,6,9] ÷ 6.48 = [0.46, 0.93, 1.39] 数值变稳定了！
        """
        # 保存原始数据类型，因为计算过程需要高精度
        orig_dtype = x.dtype

        # 转换为float32高精度，避免计算过程中的数值误差
        x = x.to(torch.float32)
        #开始计算方差，上方注释中第1，2步
        var = x.pow(2).mean(dim=-1, keepdim=True)

        #开方（RMS操作） 
        # torch.rsqrt(var + self.eps): 计算 1/√(var+eps)，即RMS的倒数
        # 为什么加eps？防止var=0时除零错误，就像数学中的"安全措施"
        # x.mul_(): 原地乘法，将x乘以RMS的倒数，实现归一化
        x.mul_(torch.rsqrt(var + self.eps))

        # 应用可学习权重并恢复原始数据类型
        # 转回原始类型，然后乘以权重参数
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 把两个操作合并，减少内存占用和计算量
        
        # 保存原始数据类型
        orig_dtype = x.dtype

        # 融合残差连接
        # 同时转换为float32并加上残差，一石二鸟
        # x.to(float32).add_(residual.to(float32)): x = x + residual
        x = x.to(torch.float32).add_(residual.to(torch.float32))

        # 保存新的残差用于下一层
        # ⚠️ 重要：必须在归一化前保存！防止原信息涵盖内容在进行RMS操作后丢失
        # 如果先归一化再保存，下一层获得的信息会丢失原始强度
        residual = x.to(orig_dtype)

        # 标准RMS归一化流程
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        
        # 返回：归一化后的输出 + 更新后的残差（供下一层使用）
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        #和之前一样第一次先没有residual
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

Rotary_emb


class Qwen3Attention(nn.Module):

    def __init__() -> None:
        
        # 旋转位置编码(RoPE)
        self.rotary_emb = get_rope(
            self.head_dim,          # 头维度: 128
            rotary_dim=self.head_dim,   # 旋转维度: 128 (全部维度参与旋转)
            max_position=max_position,  # 最大位置: 40960
            base=rope_theta,        # 基频: 1000000 (比标准值大100倍，支持更长序列)
            rope_scaling=rope_scaling,  # 缩放策略: None
        )
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)#在预处理完QK之后，进入attnention layer之前进行
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output
from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(
    x: torch.Tensor,    # 输入向量 [num_tokens, num_heads, head_dim]
    cos: torch.Tensor,  # 余弦值 [num_tokens, head_dim//2]
    sin: torch.Tensor,  # 正弦值 [num_tokens, head_dim//2]
) -> torch.Tensor:
    """
    应用旋转位置编码的核心函数
    
    数学原理：复数旋转
    将向量看作复数，通过乘以 e^(iθ) 进行旋转
    (a + bi) * e^(iθ) = (a*cos(θ) - b*sin(θ)) + (a*sin(θ) + b*cos(θ))i
    
    实现方式：
    - 将128维向量分成64对 (x1,x2), (x3,x4), ..., (x127,x128)
    - 每对看作一个复数 x1 + x2*i
    - 应用旋转变换
    """
    # 为cos和sin增加维度，使其能够广播到x的形状
    cos = cos.unsqueeze(-2)  # [num_tokens, 1, head_dim//2] 增加头维度
    sin = sin.unsqueeze(-2)  # [num_tokens, 1, head_dim//2] 增加头维度
    
    # 将输入向量分成两半，模拟复数的实部和虚部
    # 例如：128维 -> 前64维(实部) + 后64维(虚部)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    
    # 应用旋转变换公式
    # 这是复数旋转 (a + bi) * e^(iθ) 的实现
    y1 = x1 * cos - x2 * sin  # 新的实部：a*cos(θ) - b*sin(θ)
    y2 = x2 * cos + x1 * sin  # 新的虚部：a*sin(θ) + b*cos(θ)
    
    # 将实部和虚部重新拼接，恢复原始维度
    return torch.cat((y1, y2), dim=-1).to(x.dtype)  # 转回原始数据类型

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块
    
    核心作用：为每个token位置计算旋转角度，让模型知道词与词之间的距离关系
    """

    def __init__(
        self,
        head_size: int,                 # 注意力头维度，通常是128
        rotary_dim: int,                # 旋转维度，通常等于head_size
        max_position_embeddings: int,   # 最大支持的序列长度，如40960
        base: float,                    # 基础频率，如1000000
                                       # base的作用详解：
                                       # 1. 控制位置编码的"旋转速度"：base越大，旋转越慢，周期越长，可以存储的信息就越多
                                       # 2. 影响模型的序列长度能力：更大的base支持更长的序列
                                       # 3. 标准RoPE使用base=10000，Qwen3使用1000000（大100倍）
                                       # 4. 为什么用1000000？因为要支持40960长度的序列：
                                       #    - 标准base=10000适合~2048长度
                                       #    - base=1000000适合~40000+长度
                                       # 5. 数学原理：频率 = 1/(base^(2i/dim))
                                       #    base越大 → 频率越小 → 旋转越慢 → 位置区分度在更长距离内保持有效
                                       # 6. 为什么不能设为无限大或直接按序列长度计算？
                                       #    - 无限大：频率→0，所有位置旋转角度相同，失去位置区分能力
                                       #    - 按序列长度算：训练时base已固化在权重中，推理时改变=模式不匹配
                                       #    - 需要平衡：序列支持长度 vs 数值稳定性 vs 多频率层次 vs 训练一致性
    ) -> None:
        super().__init__()
        self.head_size = head_size                      # 保存头维度：128
        assert rotary_dim == head_size                  # 确保旋转维度等于头维度
        
        # 第一步：计算基础频率
        # 为每个维度对计算不同的旋转频率
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 第二步：为每个位置生成时间序列
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 第三步：计算每个位置每个维度的旋转角度
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 第四步：预计算所有角度的余弦和正弦值
        # 这样在forward时就不需要重复计算，提高效率
        cos = freqs.cos()  # 余弦值表 [max_position, rotary_dim//2]
        sin = freqs.sin()  # 正弦值表 [max_position, rotary_dim//2]
        
        # 第五步：将cos和sin拼接并注册为buffer
        # 拼接后形状：[max_position, rotary_dim] 
        # 前一半是cos值，后一半是sin值
        cache = torch.cat((cos, sin), dim=-1)
        
        # 注册为buffer：不是模型参数，但需要随模型一起保存和移动到GPU
        # persistent=False：保存模型时不包含这个buffer（可以重新计算）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile  # PyTorch 2.0编译优化，提高运行速度
    def forward(
        self,
        positions: torch.Tensor,  # 当前batch中每个token的位置索引 [num_tokens]
        query: torch.Tensor,      # Q向量 [num_tokens, num_heads, head_dim] 
        key: torch.Tensor,        # K向量 [num_tokens, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对Q和K向量应用旋转位置编码
        
        为什么只对Q和K应用，不对V应用？
        - Q·K^T负责"关系匹配"：计算词与词之间的注意力权重，需要知道相对位置关系
        - V负责"内容携带"：提供实际的语义内容，内容本身与位置无关
        - 数学角度：attention = softmax(Q·K^T / √d) · V
          位置信息影响前半部分的权重计算，不影响后半部分的内容提取
        """
        num_tokens = positions.size(0)  # 当前处理的token数量
        
        # 根据位置索引从预计算的缓存中提取对应的cos和sin值
        # positions例如：[0, 1, 2] -> 提取第0、1、2行的cos_sin值
        cos_sin = self.cos_sin_cache[positions]  # [num_tokens, rotary_dim]
        
        # 将拼接的cos_sin分离为cos和sin两部分
        cos, sin = cos_sin.chunk(2, dim=-1)  # 各自形状：[num_tokens, rotary_dim//2]
        
        # 对Query向量应用旋转位置编码
        query_shape = query.shape  # 保存原始形状用于恢复
        # 重塑为 [num_tokens, num_heads, head_dim] 确保维度匹配
        query = query.view(num_tokens, -1, self.head_size)
        # 应用旋转变换
        query = apply_rotary_emb(query, cos, sin).view(query_shape)  # 恢复原始形状
        
        # 对Key向量应用相同的旋转位置编码
        key_shape = key.shape    # 保存原始形状用于恢复
        # 重塑为 [num_tokens, num_kv_heads, head_dim] 确保维度匹配
        # 为什么要改变形状？apply_rotary_emb函数需要特定的输入格式才能正确处理
        key = key.view(num_tokens, -1, self.head_size)
        # 应用旋转变换
        # 注意：这里数据内容已经永久改变（向量已旋转，包含位置信息）！
        # 后面的view不是"撤销旋转"，而是"恢复包装盒形状"
        key = apply_rotary_emb(key, cos, sin).view(key_shape)    # 恢复原始形状
        # 类比：把处理过的食物重新装回原包装盒 - 食物变了，盒子形状一样
        
        # 返回应用了位置编码的Q和K向量
        # 现在它们包含了位置信息，可以用于计算位置感知的注意力
        return query, key

@lru_cache(1)
def get_rope(
    head_size: int,         # 注意力头维度，通常是128
    rotary_dim: int,        # 旋转维度，通常等于head_size
    max_position: int,      # 最大支持的序列长度，如40960
    base: float,            # 基础频率，如1000000，控制位置编码的旋转速度
    rope_scaling: dict | None = None,  # RoPE缩放策略，暂不支持（预留扩展）
):
    """
    RoPE工厂函数：创建旋转位置编码实例

    - 给定配置参数，返回标准化的RoPE实例
    - 使用缓存避免重复创建相同配置的对象
   
    缓存机制：
    - @lru_cache(1)确保相同参数只创建一次实例
    - 避免重复初始化，节省内存和计算时间
    """
    assert rope_scaling is None  
    
    # 创建并返回RotaryEmbedding实例
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

Flash Attention

Flash Attention
Attention计算
传统Attention计算，输入Token向量生成Q、K、V矩阵，通过Q乘以K的转置得到S，对S按行求Softmax得到注意力矩阵P，最后P乘以V得到输出O。大致计算过程如下：
Q、K、V矩阵存储在HBM中，形状为，其中为序列长度，为特征维度
从HBM加载Q、K矩阵到SRAM
计算
将S写回HBM
从HBM加载S到SRAM
计算
将P写回HBM
从HBM加载P和V到SRAM
计算
将O写回HBM
返回O
可以看到，中间变量如S和P矩阵的读写操作频繁，这些矩阵的大小随序列长度平方增长，中间矩阵占用的显存非常可观，虽然保留这些中间结果会占用显存，但在反向传播时它们是计算梯度所必需的。
在模型训练中，训练速度主要受两种因素制约：一是计算瓶颈（Compute Bound），当运算复杂度高而数据量不大时，如大型矩阵乘法和多通道卷积操作；二是其他因素。
另一种情况是Memory Bound，即训练速度的瓶颈在于HBM数据的读取速度。当从HBM读取数据的速度跟不上运算速度时，算力会处于等待数据的状态。
Attention计算主要属于Memory Bound操作。相比之下，Compute Bound操作（如矩阵乘法）耗时较少，而Memory Bound操作占据了大部分时间，所以优化Memory Bound很有必要。
主要的设计背景可以归结于以下三点：
● Transformers 计算长文本时平方时间复杂度，很耗时很耗显存
● 有改进的方法降低计算复杂度，没太大用，原因缓存存取很耗时
● FlashAttention 优化了显存存取，非优化计算复杂度



如上图所示，GPU除了HBM之外，计算实际工作都在 SRAM中，而且SRAM有19TB/S 的计算速度比HBM 1.5TB/S 快12.67倍，不过只有20MB可以使用
所以得出结论：SRAM<-->HBM为耗时瓶颈，传统Attention 要做大约7次交换，如果能直接在20MB的SRAM中实现Attention计算不做过多内存交换可以有效加速 （大约7.6x倍）
那么很自然可以想到可以通过对（Q × Kᵀ）进行分块，直接将其送到SRAM中进行计算，因为矩阵乘天然可分块，块与块之间无跨块依赖。只要把 Q 按行、K 按列tile，任何一对 (Q_i, K_j) 都能独立完成一小块乘法：
S_ij = Q_i · K_jᵀ 
算完立刻可以丢掉 S_ij，只需要把结果累加进下一步所需的统计量。
因此 GEMM 本身完全可以“算-丢-算”，不占 HBM 带宽。

但是，在实际操作中，直接这样是不可以实现的，因为在计算完QK后要进行softmax， 普通 softmax 需要先看完整行，再一次性做归一化：
s_k   = Σ_j e^{ Q_k  · K_jᵀ }          # 分母
P_kj  = e^{ Q_k·K_jᵀ } / s_k           # softmax 输出
• 行分母 s_k 依赖于 整行 所有列的指数和；
• 如果把列分块，就必须把所有 S_kj 先暂存到 HBM，
再扫一遍做归一化 → 频繁写回 / 读回，相当于没有优化。
所以关键是：让行分母可在线更新，不必一次性求完。
online-softmax
online-softmax就提供了一种可以渐进地维护一行的softmax的算法，
m = 行最大值、ℓ = 累加的未归一化分母、o = 累加的分子。
每来一块 (S_ij , V_j) 仅用这三样东西就能迭代更新。
大致算法为
#定义当前块内的局部量
m̃ = rowmax(S_ij)                         # 本块最大
Δℓ = Σ exp(S_ij - m̃)                    # 本块分母增量
Δo = Σ exp(S_ij - m̃) ⊙ V_j              # 本块分子增量

#行级递推公式：
m_new = max(m, m̃)
α      = exp(m - m_new)                  # 旧量的缩放
β      = exp(m̃ - m_new)                 # 本块的缩放

ℓ  ← α·ℓ + β·Δℓ
o  ← α·o + β·Δo
m  ← m_new

#到行块全部处理完后再一次性归一化
O_row = o / ℓ                            # 最终 softmax·V



至此，全部计算都可以直接在SRAM上完成，对于HBM可以只在输出时做读写，那一个简单的FlashAttension公式可得：
FlashAttension公式
for each row-tile  Q_i:                         # 外层循环在Flash Attention2中主要是KV在外层循环以减少o的写操作
    load Q_i → SRAM								
    m_i, ℓ_i = -∞ , 0
    O_i      = 0                               # (B_r × d) 寄存/SMEM

    for each col-tile (K_j , V_j):             # 内层循环
        load K_j , V_j → SRAM
        S_ij = Q_i  · K_jᵀ                     # tile-GEMM

        # online-softmax 更新
        m̃    = rowmax(S_ij)
        Δℓ    = rowsum( exp(S_ij - m̃) )
        Δo    = rowsum( exp(S_ij - m̃) ⊙ V_j )

        m_new = max(m_i , m̃)
        α     = exp(m_i - m_new)
        β     = exp(m̃  - m_new)

        ℓ_i   = α * ℓ_i + β * Δℓ
        O_i   = α * O_i + β * Δo
        m_i   = m_new
    end for

    O_i = O_i / ℓ_i                            # 行块归一化
    write O_i , ℓ_i → HBM                      # 只写一次
end for

图解

1. 外层循环（Outer Loop，红色箭头）
• 依次把 Kᵀ 和 V 按列 tile 成许多小块。
• 每次只把当前块复制到片上 SRAM，避免把整张 N×d 的矩阵放进 HBM⇄SRAM 反复来回。
2. 内层循环（Inner Loop，蓝色箭头）
• 对 Q 按行 tile，同样一小块一小块地读到 SRAM。
• SRAM 中的一个 Q-块和一个 K-块做局部乘法，得到局部相关性 S_ij = Q_i · K_jᵀ。
• 乘完立即进入 online-softmax 的递推：更新行最大值 m、分母 ℓ 和分子 o，不把中间 S 保存在 HBM。
3. online-softmax 使得整行 softmax 可以边算边归一化，彻底省掉了 “先把全部 S 写回 HBM → 再读回来做 softmax” 这一步骤。
4. 当一行 Q 处理完所有 K-块后，softmax(QKᵀ) 已经就地完成；再把对应的 V-块乘进来，结果 sm(QKᵀ)V 直接写回 HBM。整个流程只对最终输出进行一次写回。

在vllm中的代码
class Qwen3Attention(nn.Module):

    def __init__() -> None:
        
        # Flash Attention核心计算
        self.attn = Attention(
            self.num_heads,         # 当前GPU的注意力头数
            self.head_dim,          # 头维度: 128
            self.scaling,           # 缩放因子
            self.num_kv_heads,      # 当前GPU的KV头数
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)#在这里调用，输入是预处理完的qk和v
        output = self.o_proj(o)
        return output
# Flash Attention的核心库=：
# - flash_attn_varlen_func: 处理不同长度序列的batch（prefill专用）
# - flash_attn_with_kvcache: 利用历史计算结果的快速attention（decode专用）
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context  # 获取当前是prefill还是decode等信息


class Attention(nn.Module):
    """
    Flash Attention模块 - 自注意力机制的高效实现
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor  # 声明输出变量
        
        # 使用了GQA 优化
        # ========== GQA (Grouped Query Attention) 优化策略详解 ==========
        # 
        # 什么是GQA？
        # GQA是一种内存和计算优化的注意力机制，通过减少KV头数来提升效率
        # 
        # 传统多头注意力 vs Qwen3的GQA优化：
        # • 传统方式：16个Q头 + 16个K头 + 16个V头 = 48个矩阵
        # • GQA优化：16个Q头 + 8个K头 + 8个V头 = 32个矩阵 (减少33%参数)
        # 
        # 头映射关系：
        # • Q1, Q2 → 共享 K1, V1    • Q3, Q4 → 共享 K2, V2
        # • Q5, Q6 → 共享 K3, V3    • Q7, Q8 → 共享 K4, V4  
        # • Q9, Q10 → 共享 K5, V5   • Q11, Q12 → 共享 K6, V6
        # • Q13, Q14 → 共享 K7, V7  • Q15, Q16 → 共享 K8, V8
        # 
        # 优化效果：
        # • 内存节省：KV缓存减少50% (长序列生成的主要瓶颈)
        # • 计算加速：KV计算量减半，推理速度显著提升
        # • 性能保持：几乎无精度损失(<2%)，因为相邻Q头通常关注相似信息
        # • 扩展性好：支持更长序列和更大批次的推理
        #
        # 为什么GQA有效？
        # 研究发现在多头注意力中，很多注意力头学到的模式是相似的，
        # 因此可以让多个查询头共享相同的键值头，在几乎不损失性能的情况下大幅减少计算和内存开销
        # ================================================================
        
        # 重塑tensor形状，从平铺格式变为多头格式
        q = q.view(-1, self.num_heads, self.head_dim)      # [5, 16, 128] 分成16个查询头
        k = k.view(-1, self.num_kv_heads, self.head_dim)   # [5, 8, 128] 分成8个key头
        v = v.view(-1, self.num_kv_heads, self.head_dim)   # [5, 8, 128] 分成8个value头
        
        
        context = get_context()
        
        # 获取KV缓存的引用
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 增量存储新的K、V到缓存
        if k_cache.numel() and v_cache.numel():  # numel()>0表示缓存已初始化
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
         # prefill模式
        if context.is_prefill: 
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q,    # Q序列的最大长度，如20
                                       cu_seqlens_q=context.cu_seqlens_q,    # Q累积长度：[0,5,12,20]表示3个序列长度为5,7,8
                                       max_seqlen_k=context.max_seqlen_k,    # K序列的最大长度，通常与Q相同
                                       cu_seqlens_k=context.cu_seqlens_k,    # K累积长度数组
                                       softmax_scale=self.scale,             # 注意力缩放：1/√128≈0.088
                                       causal=True,                          # 因果掩码：只能看到当前及之前的token
                                       block_table=context.block_tables)     # 内存块映射，管理GPU内存分配
        else:    # Decode模式
            # 使用专门优化的KV缓存版本Flash Attention
            o = flash_attn_with_kvcache(q.unsqueeze(1),           # 在seq维度增加维度：[tokens,1,heads,dim]
                                        k_cache, v_cache,         # 直接使用缓存的所有历史K、V
                                        cache_seqlens=context.context_lens,  # 每个序列已缓存的长度
                                        block_table=context.block_tables,    # GPU内存块映射表
                                        softmax_scale=self.scale,            # 注意力缩放因子
                                        causal=True)                         # 保持因果性：新token不能看未来
        

        # 从多头格式 [tokens, heads, head_dim] 变回平铺格式 [tokens, heads * head_dim]
        o = o.view(-1, self.num_heads * self.head_dim)
        
        return o 
Attention Block
目前为止，我们完成了大部分Attention Block计算的解释，为了直观理解整个block的网络结构及其对应代码，Attention Block的overview如下：

def forward(
        self,
        positions: torch.Tensor,       # token位置索引，用于RoPE位置编码
        hidden_states: torch.Tensor,  # 输入隐藏状态 [batch_size, seq_len, 1024]
    ) -> torch.Tensor:
        """
        Self-Attention
        
        prefill vs decode阶段差异：
        - prefill：一次处理整个prompt，全量计算注意力矩阵 O(N²)
        - decode：只处理新生成的1个token，利用KV缓存 O(N)
        """
        # 第一步：QKV线性投影
        # 1024维 → Q(16*128) + K(8*128) + V(8*128) = 4096维
        qkv = self.qkv_proj(hidden_states)
        
        # 第二步：分离Q、K、V向量
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 第三步：Q向量归一化 
        # 操作详解：
        q_by_head = q.view(-1, self.num_heads, self.head_dim)  # 步骤1：展开多头结构
        # view作用：重新排列tensor形状，不复制数据(共享内存)
        # 为什么要展开？因为RMS归一化需要在每个头的维度上独立进行，而不是在所有头混合的维度上
        
        q_by_head = self.q_norm(q_by_head)  # 步骤2：RMS归一化
        # 对每个注意力头的向量独立做归一化，让每个头的数值稳定在合理范围内
        # 防止某些头的数值特别大压制其他头的贡献
        
        q = q_by_head.view(q.shape)  # 步骤3：恢复原始形状
        # 重新合并多头结构，恢复到原来的tensor形状，确保后续计算的兼容性
        
        # 第四步：K向量归一化（和Q向量归一化操作一致）
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        
        # 第五步：应用RoPE位置编码
        # 基频1000000支持40960长度的位置编码
        q, k = self.rotary_emb(positions, q, k)
        
        # 第六步：Flash Attention计算
        # 计算注意力权重：Q·K^T → softmax → 乘以V
        # 结果：每个token获得加权后的上下文信息
        o = self.attn(q, k, v)
        
        # 第七步：输出投影
        output = self.o_proj(o)  # 多头输出 → 1024维
        
        return output
MLP/FFN


FFN = Feed Forward Network = 前馈神经网络 = 大脑的"深度思考"模块
FFN = MLP

什么是FFN？
- 就是一个"扩展思考→筛选过滤→整合结论"的三步处理过程
- 模拟人脑从初步理解到深度分析再到得出结论的思维过程

为什么是3072维？
- 输入: 1024维 (基础理解)
- 扩展: 3072维 (3倍空间进行深度思考)  
- 输出: 1024维 (整合后的精炼结论)
- 经验法则: FFN维度通常是隐藏维度的3-4倍，提供充足"思考空间"

实际工作流程示例：
输入："我想去公园" (1024维初步理解)
↓
扩展思考 (1024→3072维)：
- "我" → [个人意愿, 主观感受, 行动主体, 情感状态...]
- "想" → [意图表达, 情感倾向, 未来规划, 愿望程度...]  
- "去" → [移动行为, 空间转换, 目的导向, 动作执行...]
- "公园" → [休闲场所, 自然环境, 社交空间, 放松地点...]
↓
门控筛选 (SiLU激活)：
- 保留有用思考: 休闲意图、移动行为、放松需求
- 过滤无关联想: 其他杂念和不相关的概念
↓
整合输出 (3072→1024维)：
- 得出结论: "用户表达了去公园放松的明确意图"

类比理解：
- 1024维像"第一印象" (快速直觉反应)
- 3072维像"仔细琢磨" (展开所有可能的理解角度)
- 最终1024维像"得出结论" (整合思考后的精准理解)
MLP代码
class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,           # 真实值: 1024 - 隐藏层维度，输入特征维度
        intermediate_size: int,     # 真实值: 3072 - 中间层维度，FFN的扩展维度(3倍思考空间)
        hidden_act: str,            # 真实值: "silu" - 激活函数类型，SiLU = Sigmoid Linear Unit
    ) -> None:
        super().__init__()
        # gate_up_proj：将输入同时映射到两个高维空间
        # 1024维 → [3072维gate, 3072维up] = 6144维总输出，用于gate机制
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,            # 输入: 1024维 (基础理解)
            [intermediate_size] * 2,# 输出: [3072, 3072] - gate门控向量 + up扩展向量
            bias=False,             # 不使用偏置项
        )
        
        # 下降投影：将高维特征映射回隐藏维度
        # 3072维深度思考结果 → 1024维精炼结论
        self.down_proj = RowParallelLinear(
            intermediate_size,      # 输入: 3072维 (深度思考结果)
            hidden_size,            # 输出: 1024维 (整合后的理解)
            bias=False,             # 不使用偏置项
        )
        
        # 激活函数验证和初始化
        assert hidden_act == "silu"  # 确保使用SiLU激活函数进行门控筛选
        self.act_fn = SiluAndMul()   # SiLU激活 + 逐元素乘法的组合 (门控机制核心)

    def forward(self, x):
        # 第一步：gate_up_proj
        # 将输入同时投影到两个高维空间：gate和up
        # 例子：输入1024维 → 分别投影到3072维，得到两个3072维向量
        # gate用于控制信息流，up用于特征变换
        gate_up = self.gate_up_proj(x)  # 形状: [total_tokens, 2 * intermediate_size] = [tokens, 6144]
        
        # 第二步：SiLU门控激活 - 用于筛选有用信息
        # 将gate_up分为两部分，应用SiLU激活函数和逐元素乘法
        # SiLU(gate) * up：门控机制让模型学会选择性地传递信息
        # gate经过SiLU激活后与up逐元素相乘，实现自适应的信息过滤
        x = self.act_fn(gate_up)  # 计算完之后还是3072维 
        
        # 第三步：下降投影 - 整合成精炼结论
        # 将高维特征映射回原始隐藏维度
        # 3072维深度思考结果 → 1024维精炼理解
        # 这些高维特征立刻影响当前token的表示，直接参与最终的logits计算
        x = self.down_proj(x)  # 计算完之后只有1024维了
        
        # 返回"深度思考"后的token表示，用于后续计算
        # 包含了丰富的非线性变换信息，影响模型的输出
        return x


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步：将输入张量在最后维度分成两半
        # 例如：[batch, 6144] → x[batch, 3072] + y[batch, 3072]
        # x作为待激活的信号，y作为门控信号
        x, y = x.chunk(2, -1)
        
        # 第二步：门控SiLU激活
        # SiLU(x) * y：对x应用SiLU激活，然后用y进行门控
        # 门控机制：y决定哪些信息通过，哪些被抑制
        return F.silu(x) * y

Decode Layer

目前为止，我们完成了几乎全部的Decode Layer计算的讲解，以下是Decode Layer的全部代码：
    def forward(
        self,
        positions: torch.Tensor,        # token位置索引，用于RoPE位置编码
        hidden_states: torch.Tensor,   # 输入的隐藏状态 [tokens, 1024]
        residual: torch.Tensor | None, # 残差连接的累积值，用于梯度优化
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        
        if residual is None:
            # 情况1：第1层处理 (residual is None)
            # 创建残差连接的"起点"：保存原始输入作为后续累积的基准
            residual = hidden_states  
            # 第一次层归一化：为自注意力做预处理，稳定训练过程
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 情况2：第2-28层处理 (residual已存在)
            # 融合残差连接：将当前输入与累积的残差相加，然后进行层归一化
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # 自注意力机制：让每个token关注序列中的其他token
        # 16头注意力，每头128维，支持40960长度的位置编码
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 第二次残差连接和层归一化：为MLP前馈网络做预处理
        # 继续累积残差：将attention的输出与residual融合
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP前馈神经网络：1024→3072→1024的非线性变换
        # 使用SiLU激活函数和门控机制增强表达能力
        hidden_states = self.mlp(hidden_states)
        
        # 返回：当前层处理后的隐藏状态 + 更新后的残差连接
        # hidden_states: 经过当前层处理的新特征表示
        # residual: 累积了当前层信息的残差连接，传递给下一层继续使用
        return hidden_states, residual
在return hidden_states 后，
    def forward(
        self,
        input_ids: torch.Tensor,    # 输入token IDs [batch_size, seq_len]
        positions: torch.Tensor,   # token位置索引 [batch_size, seq_len]  
    ) -> torch.Tensor:
        # 第一步：词嵌入 - 将token IDs转换为1024维向量表示
        # 例子：input_ids=[101, 8092, 2088] → 每个ID转为1024维向量
        hidden_states = self.embed_tokens(input_ids)  # [batch_size, seq_len, 1024]
        
        # 第二步：通过28层Transformer解码器逐层处理
        # 每层执行：RMSNorm → 自注意力 → RMSNorm → FFN，带残差连接
        
        # 🔥 关键：residual参数的跨层传递机制
        # residual = None: 初始化为空，表示还没有累积的残差连接
        residual = None  # 残差连接初始化
        
        for layer in self.layers:  # 遍历28层解码器
            # 🔥 关键：residual参数的传递过程
            # 第1层: residual=None → layer内部创建residual → 返回(new_hidden, residual)  
            # 第2-28层: 接收上一层的residual → layer内部更新residual → 返回(new_hidden, updated_residual)
            # 这样每层都能利用之前所有层的累积信息，避免梯度消失
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 第三步：最终层归一化 - 稳定模型输出
        hidden_states, _ = self.norm(hidden_states, residual)  # [batch_size, seq_len, 1024]
        
        # 返回最终的隐藏状态表示，准备用于语言建模头
        return hidden_states

def forward(self, x: torch.Tensor):
        """
        语言模型头的预测过程：从隐藏状态到词汇概率
        
        输入：隐藏状态向量（比如896维的特征向量）
        输出：词汇表上的概率分布（151936个词每个都有一个概率）
        """
        
        context = get_context()
        if context.is_prefill:
            # === prefill阶段的优化：只要最后一个位置的预测 ===
            # 在处理输入序列"我 喜欢 吃 苹果"时：
            # - prefill阶段：我们只关心最后位置（"苹果"后面）的预测
            # - 不需要预测"我"后面、"喜欢"后面、"吃"后面是什么
            # cu_seqlens_q[1:] - 1：获取每个序列的最后一个位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # === 核心计算：隐藏状态 → 词汇logits ===
        # 这是一个线性变换：x @ W^T + b
        # x: [batch_size, 896] 隐藏状态
        # self.weight: [37984, 896] 当前GPU负责的词汇权重
        # 结果: [batch_size, 37984] 当前GPU负责词汇的logits
        logits = F.linear(x, self.weight, self.bias)
        
        if self.tp_size > 1:
            # === 多GPU协作：收集所有logits并拼接成完整结果 ===
            # 只有0号GPU准备收集容器
            # 创建4个空的logits容器，准备接收各GPU的结果
            # 其他GPU不需要准备容器
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            
            # 所有GPU把自己的logits发送给0号GPU
            # GPU 0发送：词汇0-37983的logits
            # GPU 1发送：词汇37984-75967的logits  
            # GPU 2发送：词汇75968-113951的logits
            # GPU 3发送：词汇113952-151935的logits
            dist.gather(logits, all_logits, 0)
            
            # 0号GPU负责拼接：把4块logits按顺序连接起来
            # [GPU0的logits, GPU1的logits, GPU2的logits, GPU3的logits]
            # = [0-37983词汇的概率, 37984-75967词汇的概率, 75968-113951词汇的概率, 113952-151935词汇的概率]
            # 最终得到完整的151936个词汇的概率分布！
            # 其他GPU不需要处理拼接结果，设为None
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
                
        # 返回完整的词汇概率分布
        # 在0号GPU上：logits包含所有151936个词汇的概率
        # 在其他GPU上：logits为None（因为只有0号GPU需要最终结果）
        return logits
然后根据之前讲解的代码调度结构，logits会被输入到sampler中随后转化成实际token输出

TP并行

TP 并行（Tensor Parallelism，张量模型并行）是一种“层内切分”的模型并行策略。它把每一层里体积最大的张量（通常是线性层权重或注意力投影矩阵）沿着列或行“纵向”切成 n 份，分别存放到 n 张 GPU 上：
1. 前向 / 反向时，每张卡只计算自己那一份子矩阵与输入的乘法。
2. 计算完的局部结果通过 All-Reduce / All-Gather 等通信算子拼接或求和，从而得到与单卡一致的输出和梯度。
这样做有两大直接收益：
• 显存占用缩小到 1/n，使超大模型能“分片”塞进多卡；
• 子矩阵乘法可并行运行，理论上算力与吞吐可随 GPU 数近似线性增长（通信带宽允许的前提下）。
在nano vllm中，我们主要讨论的是forward时候的TP 并行。

All-Gather/All-Reduce
首先来了解一下All-Gather 和 All-Reduce的概念，这两个都用来描述分布式计算中的集合通信：
All-Reduce
● All-Reduce将所有节点的数据进行指定运算（求和、平均等），然后把计算结果分发给每个节点。每个节点最终得到相同的聚合结果，数据大小保持不变。
● 例子：
# All-Reduce：4个GPU求和
输入: GPU0:[1,2] GPU1:[3,4] GPU2:[5,6] GPU3:[7,8]
输出: 每个GPU都得到:[16,20]
1倍数据量传输 + 计算
在传输过程中进行计算，通信量等于单节点数据量

All-Gather
● All-Gather是将每个节点的数据收集起来，然后把完整的数据集合分发给所有节点。没有计算操作，纯粹是数据传输和拼接。
# All-Gather：4个GPU数据拼接  
输入: GPU0:[A] GPU1:[B] GPU2:[C] GPU3:[D]
输出: 每个GPU都得到:[A,B,C,D]
N倍数据量传输 + 拼接
需要传输完整数据，通信量是All-Reduce的N倍（N为节点数）

Row / Column Parallel
在了解完All-Gather 和 All-Reduce，我们来看一下Tensor的拆分和并行，设输入数据为X，参数为W。X的维度 = (b, s, h)，W的维度 = (h, h')。其中：
● b：batch_size，表示批量大小
● s：sequence_length，表示输入序列的长度
● h：hidden_size，表示每个token向量的维度。
● h'：参数W的hidden_size。
则每次forward的过程是：

图中所绘是b=1时的情况。
假设现在W太大，导致单卡装不下。我们需要把W切开放到不同的卡上，则我们面临两个主要问题：
● 怎么切分W。
● 切完W后，怎么做forward。
一般来说，我们可以沿着W的行（h维度），或者列（h'维度）切分W。下面我们分别介绍这两种切割办法。
我们用N来表示GPU的数量。有几块GPU，就把W按行维度切成几份。下图展示了N=2时的切割方式：

W按照行维度切开后，X的维度和它不对齐了，这可怎么做矩阵乘法呢？让我们先来看看列切分W是如何计算的
如图所示，假如W按照行维度切开后，再把X“按列切开”，就能合成一个完整的矩阵，所以我们就可以在避免使用All-Gather，只用一次All-Reduce的情况下完成计算

在nano-vllm中的TP：
 Row/Column Parallel
# nanovllm/layers/linear.py
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 0)  # tp_dim=0表示按第0维切分
        self.input_size_per_partition = input_size    # 输入维度不变
        self.output_size_per_partition = divide(output_size, self.tp_size)  # 输出维度切分
        
        # 关键：权重矩阵只存储部分列
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接计算，无需通信
        return F.linear(x, self.weight, self.bias)

# nanovllm/layers/linear.py
class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 1)  # tp_dim=1表示按第1维切分
        self.input_size_per_partition = divide(input_size, self.tp_size)  # 输入维度切分
        self.output_size_per_partition = output_size                      # 输出维度不变
        
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 关键：All-Reduce聚合所有GPU的部分结果
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

通信模式
# Row/Column Parallel对比
Column Parallel:
输入: X [完整] → 权重: W_col [部分列] → 输出: Y [部分] → 无通信

Row Parallel:  
输入: X [部分] → 权重: W_row [部分行] → 输出: Y [部分] → All-Reduce → Y [完整]

关键差异: Column无通信，Row需All-Reduce
关于Row 和 Column Parallel结合在一起计算结果的大致流程如下图所示：

MLP
# nanovllm/models/qwen3.py
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        # gate_proj + up_proj合并为一个ColumnParallel层
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False
        )
        # down_proj使用RowParallel
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)    # Column Parallel，无通信
        x = self.act_fn(gate_up)          # SiLU(gate) * up
        x = self.down_proj(x)             # Row Parallel，All-Reduce
        return x
# nanovllm/layers/linear.py
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes  # [intermediate_size, intermediate_size]
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """处理gate_proj(id=0) + up_proj(id=1)的合并权重加载"""
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
通信模式
# MLP TP并行完整流程
输入: hidden_states [batch, seq, hidden_size] (每GPU完整副本)
     ↓
gate_up_proj (ColumnParallel): 
  - GPU0: [gate_0, up_0] GPU1: [gate_1, up_1] (无通信)
     ↓
SiLU激活: SiLU(gate) * up (每GPU独立)
     ↓
down_proj (RowParallel): All-Reduce聚合
     ↓
输出: result [batch, seq, hidden_size] (每GPU完整副本)

通信开销: 仅1次All-Reduce
计算效率: gate和up合并计算，减少内存访问
大致流程如下图所示：


GQA
# nanovllm/layers/linear.py
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, 
                 total_num_kv_heads: int | None = None, bias: bool = False):
        self.total_num_heads = total_num_heads              # Q头总数32
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads  # K,V头总数8
        
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)      # 每GPU的Q头16
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size) # 每GPU的K,V头4
        
        # 合并QKV的输出维度：Q_heads + K_heads + V_heads
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """QKV权重在合并矩阵中的精确定位"""
        param_data = param.data
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
Attention中的应用
# nanovllm/models/qwen3.py
class Qwen3Attention(nn.Module):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        # QKV投影：Column Parallel，每GPU计算部分头
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 位置编码和注意力计算（每GPU独立）
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        
        # 输出投影：Row Parallel，All-Reduce聚合
        output = self.o_proj(o)
        return output
通信模式
# GQA TP并行完整流程
输入: hidden_states [batch, seq, hidden_size] (每GPU完整副本)
     ↓
QKV投影 (ColumnParallel):
  - GPU0: Q[0:16] + K[0:4] + V[0:4] (无通信)
  - GPU1: Q[16:32] + K[4:8] + V[4:8]
     ↓
注意力计算: 每GPU独立处理自己的头组 (无通信)
     ↓
输出投影 (RowParallel): All-Reduce聚合
     ↓
输出: attention_out [batch, seq, hidden_size] (每GPU完整副本)

通信开销: 仅1次All-Reduce
负载均衡: Q和KV头数不同但均匀分布到各GPU
计算效率: 注意力头完全并行，无依赖关系
大致流程如下图所示：

模型参数解析

{
  // === 为什么hidden_size=1024？ ===
  "hidden_size": 1024,
  /*
  向量维度决定表达能力：
  256维：基础词汇理解
  512维：句子语义理解  
  1024维：复杂推理能力
  2048维：高级抽象思维
  
  1024维是计算效率与能力的平衡点
  */

  // === 为什么28层？ ===
  "num_hidden_layers": 28,
  /*
  处理层次：
  1-7层：词法分析、基础语法
  8-14层：句法结构、语义关系
  15-21层：逻辑推理、上下文理解
  22-28层：复杂推理、生成策略
  
  28层确保足够的推理深度
  */

  // === 为什么16个注意力头？ ===
  "num_attention_heads": 16,
  /*
  多头分工：
  语法头(1-4)：主谓宾结构
  语义头(5-8)：词义关系
  上下文头(9-12)：长距离依赖
  推理头(13-16)：逻辑关系
  
  16头提供全面的信息捕获
  */

  // === 为什么8个KV头？GQA优化 ===
  "num_key_value_heads": 8,
  /*
  资源优化：
  传统：16Q + 16K + 16V = 48个矩阵
  GQA：16Q + 8K + 8V = 32个矩阵
  
  每2个Q头共享1个KV头
  节省33%内存，性能损失<2%
  */

  // === 为什么FFN=3072？ ===
  "intermediate_size": 3072,
  /*
  计算扩展：1024 → 3072 → 1024
  
  3倍扩展经验值：
  2倍：计算能力不足
  3倍：最佳性价比
  4倍：资源浪费
  */

  // === 为什么支持40K长度？ ===
  "max_position_embeddings": 40960,
  /*
  应用需求：
  2K：对话
  4K：文章
  8K：报告
  40K：长文档、代码库
  */

  // === 为什么rope_theta=1M？ ===
  "rope_theta": 1000000,
  /*
  位置编码调整：
  标准10000：适合4K序列
  提升到100万：支持40K长序列
  
  数值越大，远距离位置信息保持越好
  */

  // === 为什么bfloat16？ ===
  "torch_dtype": "bfloat16",
  /*
  精度权衡：
  float32：高精度，2倍内存，慢2倍
  bfloat16：够用精度，省50%内存，快2倍
  
  AI计算的最佳数据类型
  */

  // === 为什么15万词汇？ ===
  "vocab_size": 151936,
  /*
  覆盖范围：
  中英文常用词汇：8万
  多语言词汇：3万
  代码关键字：2万
  特殊符号：2万
  
  确保全面的语言覆盖
  */
}

参考文档：
https://github.com/GeeeekExplorer/nano-vllm 
https://www.zhihu.com/search?type=content&q=Prefix%20Caching
https://zhuanlan.zhihu.com/p/663932651
https://zhuanlan.zhihu.com/p/5078640012
https://mp.weixin.qq.com/s/Yv21sb3vVUt5CRvz2vuMrA
https://zhuanlan.zhihu.com/p/691038809
https://zhuanlan.zhihu.com/p/718806323





