# import torch

# from ..utils import common_functions as c_f
# from ..utils import loss_and_miner_utils as lmu
# from ..utils.module_with_records import ModuleWithRecords
# from .base_loss_wrapper import BaseLossWrapper


# class CrossBatchMemory(BaseLossWrapper, ModuleWithRecords):
#     def __init__(self, loss, embedding_size, memory_size=1024, miner=None, **kwargs):
#         super().__init__(loss=loss, **kwargs)
#         self.loss = loss
#         self.miner = miner
#         self.embedding_size = embedding_size
#         self.memory_size = memory_size
#         self.reset_queue()
#         self.add_to_recordable_attributes(
#             list_of_names=["embedding_size", "memory_size", "queue_idx"], is_stat=False
#         )

#     @staticmethod
#     def supported_losses():
#         return [
#             "AngularLoss",
#             "CircleLoss",
#             "ContrastiveLoss",
#             "GeneralizedLiftedStructureLoss",
#             "IntraPairVarianceLoss",
#             "LiftedStructureLoss",
#             "MarginLoss",
#             "MultiSimilarityLoss",
#             "NCALoss",
#             "NTXentLoss",
#             "SignalToNoiseRatioContrastiveLoss",
#             "SupConLoss",
#             "TripletMarginLoss",
#             "TupletMarginLoss",
#         ]

#     @classmethod
#     def check_loss_support(cls, loss_name):
#         if loss_name not in cls.supported_losses():
#             raise Exception(f"CrossBatchMemory not supported for {loss_name}")

#     def forward(self, embeddings, labels, indices_tuple=None, enqueue_mask=None):
#         if indices_tuple is not None and enqueue_mask is not None:
#             raise ValueError("indices_tuple and enqueue_mask are mutually exclusive")
#         if enqueue_mask is not None:
#             assert len(enqueue_mask) == len(embeddings)
#         else:
#             assert len(embeddings) <= len(self.embedding_memory)
#         self.reset_stats()
#         device = embeddings.device
#         labels = c_f.to_device(labels, device=device)
#         self.embedding_memory = c_f.to_device(
#             self.embedding_memory, device=device, dtype=embeddings.dtype
#         )
#         self.label_memory = c_f.to_device(
#             self.label_memory, device=device, dtype=labels.dtype
#         )

#         if enqueue_mask is not None:
#             emb_for_queue = embeddings[enqueue_mask]
#             labels_for_queue = labels[enqueue_mask]
#             embeddings = embeddings[~enqueue_mask]
#             labels = labels[~enqueue_mask]
#             do_remove_self_comparisons = False
#         else:
#             emb_for_queue = embeddings
#             labels_for_queue = labels
#             do_remove_self_comparisons = True

#         queue_batch_size = len(emb_for_queue)
#         self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

#         if not self.has_been_filled:
#             E_mem = self.embedding_memory[: self.queue_idx]
#             L_mem = self.label_memory[: self.queue_idx]
#         else:
#             E_mem = self.embedding_memory
#             L_mem = self.label_memory

#         indices_tuple = self.create_indices_tuple(
#             embeddings,
#             labels,
#             E_mem,
#             L_mem,
#             indices_tuple,
#             do_remove_self_comparisons,
#         )
#         loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)
#         return loss

#     def add_to_memory(self, embeddings, labels, batch_size):
#         self.curr_batch_idx = (
#             torch.arange(
#                 self.queue_idx, self.queue_idx + batch_size, device=labels.device
#             )
#             % self.memory_size
#         )
#         self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
#         self.label_memory[self.curr_batch_idx] = labels.detach()
#         prev_queue_idx = self.queue_idx
#         self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
#         if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
#             self.has_been_filled = True

#     def create_indices_tuple(
#         self,
#         embeddings,
#         labels,
#         E_mem,
#         L_mem,
#         input_indices_tuple,
#         do_remove_self_comparisons,
#     ):
#         if self.miner:
#             indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
#         else:
#             indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)

#         if do_remove_self_comparisons:
#             indices_tuple = lmu.remove_self_comparisons(
#                 indices_tuple, self.curr_batch_idx, self.memory_size
#             )

#         if input_indices_tuple is not None:
#             if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
#                 input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
#             elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
#                 input_indices_tuple = lmu.convert_to_triplets(
#                     input_indices_tuple, labels
#                 )
#             indices_tuple = c_f.concatenate_indices_tuples(
#                 indices_tuple, input_indices_tuple
#             )

#         return indices_tuple

#     def reset_queue(self):
#         self.register_buffer(
#             "embedding_memory", torch.zeros(self.memory_size, self.embedding_size)
#         )
#         self.register_buffer("label_memory", torch.zeros(self.memory_size).long())
#         self.has_been_filled = False
#         self.queue_idx = 0


import torch
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from ..utils.module_with_records import ModuleWithRecords
from .base_loss_wrapper import BaseLossWrapper


class CrossBatchMemory(BaseLossWrapper, ModuleWithRecords):
    """
    跨批次记忆库（Cross-Batch Memory）

    核心功能：
    1. 维护一个固定大小的循环队列，存储历史批次的嵌入特征和标签
    2. 当前批次可以与历史批次中的样本构成训练对（正样本对/负样本对）
    3. 通过时间衰减机制，给予新鲜样本更高的权重，旧样本权重逐渐衰减

    工作流程：
    ┌─────────────┐
    │ 当前批次     │ → 提取特征 → 入队（更新时间戳）
    │ (Batch)     │              ↓
    └─────────────┘         ┌──────────────┐
                            │ 记忆库队列    │
                            │ [E₁,E₂,...,Eₙ]│ → 应用时间衰减权重
                            │ [L₁,L₂,...,Lₙ]│ → w_j = exp(-λΔt)
                            │ [T₁,T₂,...,Tₙ]│
                            └──────────────┘
                                    ↓
                            ┌──────────────┐
                            │ Miner挖掘    │ → 找困难样本对
                            │ Loss计算     │ → 应用双重权重
                            └──────────────┘

    关键参数说明：
    - memory_size: 队列容量（默认1024，本项目中设置为16384）
    - decay_lambda: 时间衰减系数λ（默认0.01，本项目中设置为0.05）
    - collect_stats: 是否收集负样本统计信息（用于分析实验）
    """

    def __init__(
        self,
        loss,
        embedding_size,
        memory_size=1024,
        miner=None,
        decay_lambda=0.01,  # 时间衰减系数λ，控制权重衰减速度
        collect_stats=False,  # 是否收集统计信息（用于实验分析）
        collect_interval=100,  # 统计信息收集间隔（单位：迭代次数）
        **kwargs,
    ):
        """
        初始化跨批次记忆库

        Args:
            loss: 损失函数对象（如MultiSimilarityLoss）
            embedding_size: 嵌入特征的维度（如DINOv2的768维）
            memory_size: 记忆库容量，能存储多少个历史样本（默认1024）
            miner: 困难样本挖掘器（如MultiSimilarityMiner）
            decay_lambda: 时间衰减系数λ
                - λ越大，旧样本衰减越快
                - 权重计算公式：w_j = exp(-λ * Δt)
                - 示例：λ=0.05时，100步前的样本权重≈0.0067
            collect_stats: 是否启用统计信息收集
                - True时，每隔collect_interval步收集负样本分析数据
                - 用于实验：分析负样本年龄、相似度、权重分布
            collect_interval: 收集统计信息的间隔步数
        """
        super().__init__(loss=loss, **kwargs)

        # ========== 核心组件 ==========
        self.loss = loss  # 损失函数（MultiSimilarityLoss等）
        self.miner = miner  # 困难样本挖掘器（MultiSimilarityMiner等）
        self.embedding_size = embedding_size  # 嵌入维度（如768）
        self.memory_size = memory_size  # 记忆库容量（如16384）
        self.decay_lambda = decay_lambda  # 时间衰减系数λ（如0.05）

        # ========== 统计信息收集（实验分析用） ==========
        self.collect_stats = collect_stats  # 是否启用统计收集
        self.collect_interval = collect_interval  # 收集间隔（如100步）
        self.stats_history = []  # 存储所有收集的统计数据

        # ========== 记忆库初始化 ==========
        # 初始化三个内存缓冲区：嵌入、标签、时间戳
        self.reset_queue()

        # ========== 全局时间步计数器 ==========
        # 用于计算样本"年龄" Δt = current_step - timestamp
        # 注册为buffer使得在GPU/CPU间自动移动，且保存checkpoint时包含
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

        # ========== 记录属性注册 ==========
        # 将关键参数注册为可记录属性（用于日志和监控）
        self.add_to_recordable_attributes(
            list_of_names=[
                "embedding_size",  # 嵌入维度
                "memory_size",  # 队列容量
                "queue_idx",  # 当前队列指针
                "decay_lambda",  # 衰减系数
                "collect_stats",  # 统计开关
            ],
            is_stat=False,  # 标记为配置参数而非统计量
        )

    @staticmethod
    def supported_losses():
        """
        返回支持的损失函数列表

        CrossBatchMemory作为损失包装器，可以与多种损失函数配合使用
        本项目主要使用MultiSimilarityLoss

        Returns:
            list: 支持的损失函数类名列表
        """
        return [
            "AngularLoss",
            "CircleLoss",
            "ContrastiveLoss",
            "GeneralizedLiftedStructureLoss",
            "IntraPairVarianceLoss",
            "LiftedStructureLoss",
            "MarginLoss",
            "MultiSimilarityLoss",  # ← 本项目使用的损失函数
            "NCALoss",
            "NTXentLoss",
            "SignalToNoiseRatioContrastiveLoss",
            "SupConLoss",
            "TripletMarginLoss",
            "TupletMarginLoss",
        ]

    @classmethod
    def check_loss_support(cls, loss_name):
        """
        检查给定的损失函数是否被支持

        Args:
            loss_name: 损失函数类名（字符串）

        Raises:
            Exception: 如果损失函数不在支持列表中
        """
        if loss_name not in cls.supported_losses():
            raise Exception(f"CrossBatchMemory not supported for {loss_name}")

    def reset_queue(self):
        """
        重置记忆库队列（初始化或清空）

        创建三个核心内存缓冲区：
        1. embedding_memory: 存储历史嵌入特征 [memory_size, embedding_dim]
        2. label_memory: 存储对应的标签 [memory_size]
        3. timestamp_memory: 存储进入队列的时间步 [memory_size]

        内存管理策略：
        - 使用循环队列（ring buffer）机制
        - queue_idx指向下一个待写入位置
        - 当队列满时，新样本覆盖最旧样本

        示例：
        memory_size=4, 当前状态：
        Index:      [0, 1, 2, 3]
        Embedding:  [E₁, E₂, ?, ?]  (? 表示未填充)
        Timestamp:  [10, 15, 0, 0]
        queue_idx = 2  (下次写入位置)
        has_been_filled = False  (队列未满)
        """
        # ===== 嵌入特征内存 =====
        # 形状: [memory_size, embedding_dim]
        # 例如: [16384, 768] 可存储16384个DINOv2特征
        self.register_buffer(
            "embedding_memory", torch.zeros(self.memory_size, self.embedding_size)
        )

        # ===== 标签内存 =====
        # 形状: [memory_size]
        # 存储每个样本的类别/地点ID
        self.register_buffer("label_memory", torch.zeros(self.memory_size).long())

        # ===== 时间戳内存 =====
        # 形状: [memory_size]
        # 存储样本进入队列的时间步（用于计算年龄）
        self.register_buffer("timestamp_memory", torch.zeros(self.memory_size).long())

        # ===== 队列状态标记 =====
        self.has_been_filled = False  # 队列是否已填满过（首次循环覆盖前为False）
        self.queue_idx = 0  # 循环队列指针（指向下一个写入位置）

    def add_to_memory(self, embeddings, labels, batch_size):
        """
        将当前批次样本添加到记忆库（循环队列更新）

        工作流程：
        1. 计算当前批次在队列中的位置索引（处理循环覆盖）
        2. 更新三个内存：嵌入、标签、时间戳
        3. 移动队列指针到下一个待写入位置
        4. 检测队列是否首次填满

        Args:
            embeddings: 当前批次的嵌入特征 [batch_size, embedding_dim]
            labels: 当前批次的标签 [batch_size]
            batch_size: 批次大小

        示例：
        memory_size=4, queue_idx=2, batch_size=3

        步骤1: 计算索引
        curr_batch_idx = [2, 3, 0]  (模运算处理循环)
                          ↑  ↑  ↑
                          新  新  覆盖旧样本

        步骤2: 更新内存
        embedding_memory[2] = emb_0
        embedding_memory[3] = emb_1
        embedding_memory[0] = emb_2  ← 覆盖最旧样本

        步骤3: 更新指针
        queue_idx: 2 → 1  (下次从索引1开始写入)
        has_been_filled: False → True  (检测到循环覆盖)
        """
        # ===== 步骤1: 计算当前批次的内存位置索引 =====
        # 使用模运算实现循环队列
        # 例如: queue_idx=16380, batch_size=8, memory_size=16384
        #       索引范围 = [16380, 16381, ..., 16383, 0, 1, 2, 3]
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )

        # ===== 步骤2: 同步更新三个内存 =====
        curr_step = self.current_step.item()  # 获取当前全局时间步

        # 更新嵌入特征（使用detach避免梯度传播到历史样本）
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()

        # 更新标签
        self.label_memory[self.curr_batch_idx] = labels.detach()

        # 更新时间戳（记录当前时间步，用于后续计算年龄）
        self.timestamp_memory[self.curr_batch_idx] = curr_step

        # ===== 步骤3: 移动队列指针 =====
        prev_queue_idx = self.queue_idx  # 保存旧指针用于检测循环
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size

        # ===== 步骤4: 检测队列首次填满 =====
        # 判断条件：指针发生"回绕"（新指针 <= 旧指针）
        # 例如: prev=16380, new=4 → 发生回绕 → 队列已填满
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True
            # 填满后，所有memory_size个位置都可用于训练

    def compute_time_decay_weights(self, device):
        """
        计算时间衰减权重（核心算法）

        数学公式：
            w_j = exp(-λ * Δt_j)

        其中：
        - w_j: 第j个样本的权重
        - λ: 衰减系数（decay_lambda，如0.05）
        - Δt_j: 样本年龄 = current_step - timestamp_j

        物理意义：
        - Δt=0 (刚入队): w=exp(0)=1.0 → 满权重
        - Δt=10 (10步前): w=exp(-0.05×10)≈0.606 → 权重降至60%
        - Δt=50: w=exp(-0.05×50)≈0.082 → 权重降至8%
        - Δt=100: w=exp(-0.05×100)≈0.0067 → 权重降至0.67%

        设计目的：
        1. 新鲜样本主导训练：近期样本与当前模型状态更一致
        2. 平滑过渡：指数函数避免阶跃式权重变化
        3. 自动过滤陈旧样本：旧样本权重接近0，自动被忽略

        Args:
            device: 计算设备（cuda或cpu）

        Returns:
            torch.Tensor: 权重向量 [memory_size]，范围(0, 1]

        示例：
        假设 memory_size=4, current_step=100, λ=0.05
        timestamp_memory = [95, 90, 80, 50]  (各样本入队时间步)

        计算过程：
        Δt = [5, 10, 20, 50]  (年龄)
        weights = [exp(-0.25), exp(-0.5), exp(-1.0), exp(-2.5)]
                = [0.779, 0.606, 0.368, 0.082]

        解释：
        - 5步前的样本保留78%权重（主导训练）
        - 50步前的样本仅8%权重（影响微弱）
        """
        # ===== 步骤1: 获取当前全局时间步 =====
        curr_step = self.current_step.item()

        # ===== 步骤2: 计算所有样本的年龄 Δt =====
        # 年龄 = 当前时间 - 样本入队时间
        # 形状: [memory_size]
        delta_t = curr_step - self.timestamp_memory.to(device)

        # ===== 步骤3: 应用指数衰减公式 =====
        # w_j = exp(-λ * Δt_j)
        # 使用float()确保数值稳定性
        weights = torch.exp(-self.decay_lambda * delta_t.float())

        # ===== 步骤4: 全局时间步递增 =====
        # 每次计算权重后，时间向前推进一步
        self.current_step += 1

        return weights  # 形状: [memory_size], 值域: (0, 1]

    def collect_negative_sample_stats(
        self, embeddings, labels, indices_tuple, E_mem, L_mem, time_weights
    ):
        """
        收集负样本统计信息（实验分析工具）

        分析目标：
        1. 负样本来源分布：当前批次 vs 记忆库
        2. 记忆库负样本的年龄分布（多久前的样本被选为困难负样本）
        3. 负样本相似度分布（困难程度量化）
        4. 时间衰减权重的实际值（衰减效果验证）

        收集数据用途：
        - 可视化分析：15_analyze_negative_samples.ipynb
        - 验证假设：新鲜困难负样本是否主导训练
        - 优化参数：调整decay_lambda和memory_size

        Args:
            embeddings: 当前批次嵌入 [batch_size, embedding_dim]
            labels: 当前批次标签 [batch_size]
            indices_tuple: Miner挖掘的索引元组
                - MultiSimilarityMiner: (a1_idx, p_idx, a2_idx, n_idx)
                - TripletMiner: (anchor_idx, pos_idx, neg_idx)
            E_mem: 记忆库嵌入特征
            L_mem: 记忆库标签
            time_weights: 时间衰减权重 [memory_size]

        收集策略：
        - 间隔采样：每collect_interval步收集一次（避免性能影响）
        - 完整记录：年龄、相似度、权重三个维度
        - 分类统计：区分当前批次负样本和记忆库负样本
        """
        # ===== 步骤1: 间隔控制（避免频繁统计影响训练速度） =====
        curr_step = self.current_step.item()
        if curr_step % self.collect_interval != 0:
            return  # 不在收集间隔点，直接返回

        # ===== 步骤2: 解析索引元组（适配不同Miner格式） =====
        if len(indices_tuple) == 4:
            # MultiSimilarityMiner返回4元组
            # (a1, p, a2, n): a1与p配对，a2与n配对
            a1_idx, p_idx, a2_idx, n_idx = indices_tuple
            anchor_idx = a2_idx  # 使用与负样本配对的anchor
        elif len(indices_tuple) == 3:
            # TripletMiner返回3元组
            # (a, p, n): anchor与positive和negative配对
            anchor_idx, p_idx, n_idx = indices_tuple
        else:
            return  # 不支持的格式

        # ===== 步骤3: 计算负样本相似度（困难程度量化） =====
        # 提取anchor样本的嵌入
        anchor_embs = embeddings[anchor_idx]  # [num_pairs, dim]

        # 合并当前批次和记忆库（负样本可能来自两者）
        # 索引约定：[0, batch_size)为当前批次，[batch_size, ∞)为记忆库
        negative_embs = torch.cat([embeddings, E_mem], dim=0)
        neg_embs = negative_embs[n_idx]  # [num_pairs, dim]

        # 计算余弦相似度（detach避免梯度问题）
        # 值域[-1, 1]，越接近1表示越相似（越困难的负样本）
        cosine_sim = (
            torch.nn.functional.cosine_similarity(anchor_embs, neg_embs, dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        # ===== 步骤4: 区分负样本来源（当前批次 vs 记忆库） =====
        batch_size = len(embeddings)
        # 索引 >= batch_size 的样本来自记忆库
        is_from_memory = (n_idx >= batch_size).detach().cpu().numpy()

        # ===== 步骤5: 统计记忆库负样本的详细信息 =====
        memory_mask = is_from_memory
        # 将全局索引转换为记忆库内部索引
        memory_neg_idx = n_idx[memory_mask] - batch_size

        if len(memory_neg_idx) > 0:
            # 提取时间戳并计算年龄
            neg_timestamps = self.timestamp_memory[memory_neg_idx].cpu().numpy()
            ages = curr_step - neg_timestamps  # 样本年龄（单位：迭代次数）

            # 提取对应的时间衰减权重
            weights = time_weights[memory_neg_idx].detach().cpu().numpy()

            # ===== 步骤6: 组织统计数据 =====
            stats_entry = {
                "iteration": curr_step,  # 当前训练步数
                # 总体统计
                "num_total_negatives": len(n_idx),  # 总负样本数
                "num_memory_negatives": int(memory_mask.sum()),  # 来自记忆库的数量
                "num_batch_negatives": int((~memory_mask).sum()),  # 来自当前批次的数量
                # 记忆库负样本详细信息（用于分析）
                "memory_negative_ages": ages.tolist(),  # 年龄列表
                "memory_negative_similarities": cosine_sim[
                    memory_mask
                ].tolist(),  # 相似度
                "memory_negative_weights": weights.tolist(),  # 时间衰减权重
                # 当前批次负样本信息（对比用）
                "batch_negative_similarities": (
                    cosine_sim[~memory_mask].tolist() if (~memory_mask).any() else []
                ),
            }

            # 添加到历史记录
            self.stats_history.append(stats_entry)

            # ===== 步骤7: 周期性打印统计摘要 =====
            if curr_step % (self.collect_interval * 10) == 0:
                print(f"\n[Iteration {curr_step}] Negative Sample Statistics:")
                print(f"  Total negatives: {len(n_idx)}")
                print(
                    f"  From memory: {memory_mask.sum()} "
                    f"({100*memory_mask.sum()/len(n_idx):.1f}%)"
                )
                print(f"  Avg age (memory): {ages.mean():.1f} iterations")
                print(
                    f"  Avg similarity (memory): {cosine_sim[memory_mask].mean():.4f}"
                )
                print(f"  Avg weight (memory): {weights.mean():.4f}")

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_mask=None):
        """
        前向传播（主要流程）

        完整工作流程：
        ┌─────────────────────────────────────────────────────┐
        │ 输入: 当前批次 (embeddings, labels)                 │
        └────────────────┬────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────────────────┐
        │ 步骤1: 数据预处理                                  │
        │ - 设备同步 (CPU/GPU)                               │
        │ - 分离入队样本和参与训练样本 (若使用enqueue_mask) │
        └────────────────┬───────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────────────────┐
        │ 步骤2: 更新记忆库                                  │
        │ - add_to_memory(): 当前批次入队                   │
        │ - 更新embedding/label/timestamp三个内存           │
        └────────────────┬───────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────────────────┐
        │ 步骤3: 计算时间衰减权重                            │
        │ - compute_time_decay_weights()                    │
        │ - w_j = exp(-λΔt) for all samples in memory      │
        └────────────────┬───────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────────────────┐
        │ 步骤4: 困难样本挖掘                                │
        │ - Miner挖掘: 当前批次 × 记忆库 → 困难样本对       │
        │ - 去除自比较 (若需要)                             │
        └────────────────┬───────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────────────────┐
        │ 步骤5: 统计信息收集 (可选)                         │
        │ - collect_negative_sample_stats()                │
        │ - 分析负样本年龄/相似度/权重分布                   │
        └────────────────┬───────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────────────────┐
        │ 步骤6: 计算损失                                    │
        │ - MultiSimilarityLoss接收time_weights            │
        │ - 应用双重权重: 困难度权重 × 时间衰减权重          │
        └────────────────┬───────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────────────────┐
        │ 输出: (loss, indices_tuple)                        │
        └────────────────────────────────────────────────────┘

        Args:
            embeddings: 当前批次嵌入 [batch_size, embedding_dim]
            labels: 当前批次标签 [batch_size]
            indices_tuple: 预定义的样本对索引（可选，通常为None让Miner挖掘）
            enqueue_mask: 布尔掩码，指定哪些样本入队 (可选)
                - None: 所有样本既入队又参与训练
                - 提供掩码: True样本仅入队，False样本仅训练

        Returns:
            loss: 计算得到的损失值
            indices_tuple: 使用的样本对索引（用于监控）

        设计特点：
        1. 双重加权机制：
           - Miner筛选困难样本（基于相似度）
           - 时间衰减调整历史样本权重

        2. 自动内存管理：
           - 循环队列避免内存溢出
           - 旧样本自动被新样本覆盖

        3. 灵活的入队策略：
           - enqueue_mask=None: 所有样本入队（默认）
           - enqueue_mask提供: 分离入队和训练样本
        """
        # ========== 步骤1: 参数验证和预处理 ==========
        # 互斥性检查：indices_tuple和enqueue_mask不能同时使用
        if indices_tuple is not None and enqueue_mask is not None:
            raise ValueError("indices_tuple and enqueue_mask are mutually exclusive")

        # 批次大小检查
        if enqueue_mask is not None:
            assert len(enqueue_mask) == len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)

        self.reset_stats()  # 重置统计信息（用于记录）

        # 设备同步：将所有张量移到相同设备（GPU或CPU）
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self.label_memory = c_f.to_device(
            self.label_memory, device=device, dtype=labels.dtype
        )
        self.timestamp_memory = c_f.to_device(
            self.timestamp_memory, device=device, dtype=torch.long
        )

        # ========== 步骤2: 分离入队样本和训练样本 ==========
        if enqueue_mask is not None:
            # 模式A：部分样本入队，部分样本参与训练
            # 应用场景：数据增强时，只让原始样本入队
            emb_for_queue = embeddings[enqueue_mask]
            labels_for_queue = labels[enqueue_mask]
            embeddings = embeddings[~enqueue_mask]  # 剩余样本用于训练
            labels = labels[~enqueue_mask]
            do_remove_self_comparisons = False  # 不需要去除自比较
        else:
            # 模式B（默认）：所有样本既入队又参与训练
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True  # 需要去除自比较

        # ========== 步骤3: 更新记忆库 ==========
        # 将当前批次加入循环队列（自动更新时间戳）
        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        # ========== 步骤4: 计算时间衰减权重 ==========
        # 核心创新：为所有记忆库样本计算基于年龄的权重
        # w_j = exp(-λΔt_j)，新样本权重高，旧样本权重低
        time_weights = self.compute_time_decay_weights(device)

        # ========== 步骤5: 准备有效记忆库范围 ==========
        # 对未填满的队列，只使用已填充部分
        weighted_embedding_memory = self.embedding_memory

        if not self.has_been_filled:
            # 队列未满：只使用[0, queue_idx)范围
            E_mem = weighted_embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
            time_weights = time_weights[: self.queue_idx]
        else:
            # 队列已满：使用全部memory_size个样本
            E_mem = weighted_embedding_memory
            L_mem = self.label_memory
            time_weights = time_weights

        # ========== 步骤6: 困难样本挖掘 ==========
        # 生成训练样本对的索引（当前批次 × 记忆库）
        indices_tuple = self.create_indices_tuple(
            embeddings,
            labels,
            E_mem,
            L_mem,
            indices_tuple,
            do_remove_self_comparisons,
        )

        # ========== 步骤7: 统计信息收集（可选，用于实验分析） ==========
        if self.collect_stats and self.has_been_filled:
            # 收集负样本的年龄、相似度、权重等统计信息
            # 用于验证假设：新鲜困难负样本主导训练
            self.collect_negative_sample_stats(
                embeddings, labels, indices_tuple, E_mem, L_mem, time_weights
            )

        # ========== 步骤8: 计算损失 ==========
        if self.loss.__class__.__name__ == "MultiSimilarityLoss":
            # MultiSimilarityLoss支持时间权重
            # 在损失函数内部应用: loss_weight = exp(βs) * time_weight
            self.loss.weight = time_weights
            loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)
        else:
            # 其他损失函数（不使用时间权重）
            loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)

        return loss, indices_tuple

    def create_indices_tuple(
        self,
        embeddings,
        labels,
        E_mem,
        L_mem,
        input_indices_tuple,
        do_remove_self_comparisons,
    ):
        """
        创建样本对索引元组（用于损失计算）

        功能：
        1. 使用Miner挖掘困难样本对（anchor-positive, anchor-negative）
        2. 去除自比较对（避免样本与自己配对）
        3. 合并外部提供的索引（如果有）

        工作流程：
        ┌──────────────────────────────────────────────────┐
        │ 输入：                                            │
        │ - embeddings: 当前批次 [batch_size, dim]         │
        │ - E_mem: 记忆库 [memory_size, dim]               │
        │ - labels, L_mem: 对应标签                        │
        └────────────────┬─────────────────────────────────┘
                         ↓
        ┌──────────────────────────────────────────────────┐
        │ 步骤1: Miner挖掘困难样本对                        │
        │ - 计算相似度矩阵: embeddings × E_mem^T           │
        │ - 应用挖掘策略（如MultiSimilarityMiner）          │
        │ → 输出索引元组                                   │
        └────────────────┬─────────────────────────────────┘
                         ↓
        ┌──────────────────────────────────────────────────┐
        │ 步骤2: 去除自比较（可选）                         │
        │ - 移除样本与自己配对的情况                        │
        │ - 场景：当前批次样本既入队又参与训练时             │
        └────────────────┬─────────────────────────────────┘
                         ↓
        ┌──────────────────────────────────────────────────┐
        │ 步骤3: 合并外部索引（可选）                       │
        │ - 如果用户提供预定义索引，与Miner结果合并         │
        │ - 自动处理格式转换（triplet ↔ pairs）            │
        └────────────────┬─────────────────────────────────┘
                         ↓
        ┌──────────────────────────────────────────────────┐
        │ 输出：indices_tuple                               │
        │ - Triplet: (anchor_idx, pos_idx, neg_idx)        │
        │ - Pairs: (a1_idx, p_idx, a2_idx, n_idx)          │
        └──────────────────────────────────────────────────┘

        Args:
            embeddings: 当前批次嵌入 [batch_size, dim]
            labels: 当前批次标签 [batch_size]
            E_mem: 记忆库嵌入 [effective_memory_size, dim]
            L_mem: 记忆库标签 [effective_memory_size]
            input_indices_tuple: 用户预定义的索引（可选）
            do_remove_self_comparisons: 是否去除自比较
                - True: 当前批次样本已入队，需要避免自己与自己配对
                - False: 入队样本与训练样本不同，无需去除

        Returns:
            indices_tuple: 样本对索引
                - 长度为3: (anchor_idx, pos_idx, neg_idx) - Triplet格式
                - 长度为4: (a1_idx, p_idx, a2_idx, n_idx) - Pairs格式

        索引约定：
        - [0, batch_size): 当前批次的索引
        - [batch_size, batch_size + memory_size): 记忆库的索引

        示例：
        batch_size=4, memory_size=8
        Miner输出：
            anchor_idx = [0, 1, 2]  (当前批次)
            neg_idx = [5, 6, 7]     (记忆库，实际对应memory[1,2,3])

        自比较检测（如果样本0在memory[4]位置）：
            移除 (anchor=0, neg=4) 这样的配对
        """
        # ========== 步骤1: 困难样本挖掘 ==========
        if self.miner:
            # 使用Miner挖掘困难样本对
            # Miner内部会：
            # 1. 计算相似度矩阵
            # 2. 应用挖掘策略（如相似度阈值）
            # 3. 返回困难样本对的索引
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            # 无Miner：返回所有可能的样本对
            # 策略：所有同类为正样本对，所有异类为负样本对
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)

        # ========== 步骤2: 去除自比较 ==========
        if do_remove_self_comparisons:
            # 场景：当前批次样本已经入队到记忆库
            # 问题：样本可能与自己配对（anchor与neg是同一个样本）
            # 解决：移除这些自比较对
            #
            # curr_batch_idx: 当前批次在记忆库中的位置
            # memory_size: 用于计算记忆库索引偏移
            indices_tuple = lmu.remove_self_comparisons(
                indices_tuple, self.curr_batch_idx, self.memory_size
            )

        # ========== 步骤3: 合并外部索引（可选） ==========
        if input_indices_tuple is not None:
            # 用户提供了预定义的索引，需要与Miner结果合并

            # 处理格式不匹配：自动转换
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                # Miner返回Pairs格式，用户提供Triplet格式
                # 将用户的Triplet转换为Pairs
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                # Miner返回Triplet格式，用户提供Pairs格式
                # 将用户的Pairs转换为Triplet
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )

            # 合并两组索引（拼接）
            indices_tuple = c_f.concatenate_indices_tuples(
                indices_tuple, input_indices_tuple
            )

        return indices_tuple

    def save_stats(self, filepath):
        """
        保存统计数据到文件

        将收集的所有统计信息（stats_history）保存为pickle文件
        用于后续分析和可视化

        Args:
            filepath: 保存路径（建议使用.pkl扩展名）

        使用示例：
            # 训练完成后
            memory_bank.save_stats("logs/negative_sample_stats.pkl")

            # 在notebook中加载
            import pickle
            with open("logs/negative_sample_stats.pkl", "rb") as f:
                stats = pickle.load(f)

            # 分析数据
            for entry in stats:
                print(f"Iteration {entry['iteration']}")
                print(f"  Avg age: {np.mean(entry['memory_negative_ages'])}")
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.stats_history, f)
        print(f"Statistics saved to {filepath}")
        print(f"Total entries: {len(self.stats_history)}")
