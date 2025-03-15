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
    def __init__(
        self, 
        loss, 
        embedding_size, 
        memory_size=1024, 
        miner=None, 
        decay_lambda=0.01,  # 新增：时间衰减系数λ，控制权重衰减速度
        **kwargs
    ):
        """
        Args:
            decay_lambda: 时间衰减系数，越大表示旧样本衰减越快
            其他参数保持原始含义不变
        """
        super().__init__(loss=loss, **kwargs)
        self.loss = loss
        self.miner = miner
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.decay_lambda = decay_lambda

        print('CrossBatchMemory init...')
        
        # 初始化内存队列
        self.reset_queue()
        
        # 注册全局时间步计数器（用于计算时间差）
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))
        
        # 将新参数加入可记录属性
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "memory_size", "queue_idx", "decay_lambda"], 
            is_stat=False
        )

    @staticmethod
    def supported_losses():
        """保持原始支持的损失函数列表不变"""
        return [
            "AngularLoss", "CircleLoss", "ContrastiveLoss",
            "GeneralizedLiftedStructureLoss", "IntraPairVarianceLoss",
            "LiftedStructureLoss", "MarginLoss", "MultiSimilarityLoss",
            "NCALoss", "NTXentLoss", "SignalToNoiseRatioContrastiveLoss",
            "SupConLoss", "TripletMarginLoss", "TupletMarginLoss",
        ]
    
    @classmethod
    def check_loss_support(cls, loss_name):
        if loss_name not in cls.supported_losses():
            raise Exception(f"CrossBatchMemory not supported for {loss_name}")

    def reset_queue(self):
        """重置内存队列，新增时间戳存储"""
        # 嵌入内存：存储历史嵌入特征
        self.register_buffer(
            "embedding_memory", 
            torch.zeros(self.memory_size, self.embedding_size)
        )
        # 标签内存：存储对应的标签
        self.register_buffer("label_memory", torch.zeros(self.memory_size).long())
        # 时间戳内存：存储嵌入进入队列的时间步（新增）
        self.register_buffer(
            "timestamp_memory", 
            torch.zeros(self.memory_size).long()
        )
        self.has_been_filled = False  # 标记队列是否已填满
        self.queue_idx = 0  # 当前队列指针

    def add_to_memory(self, embeddings, labels, batch_size):
        """
        更新内存队列（修改：同时更新时间戳）
        """
        # 生成当前批次在内存中的位置索引
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )
        
        # 更新三个内存（嵌入、标签、时间戳）
        curr_step = self.current_step.item()
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        self.timestamp_memory[self.curr_batch_idx] = curr_step  # 记录当前时间步
        
        # 更新队列指针和时间步
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        self.current_step += 1  # 全局时间步递增
        
        # 检测队列是否首次填满
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def compute_time_decay_weights(self, device):
        """
        计算时间衰减权重（新增方法）
        返回形状为(memory_size,)的权重张量
        """
        # 获取当前时间步
        curr_step = self.current_step.item()
        
        # 计算时间差 Δt = current_step - timestamp
        delta_t = curr_step - self.timestamp_memory.to(device)
        
        # 计算指数衰减权重：w_j = exp(-λ * Δt)
        weights = torch.exp(-self.decay_lambda * delta_t.float())
        
        return weights

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_mask=None):
        """前向传播（修改：应用时间衰减权重）"""
        # 参数检查（保持原始逻辑）
        if indices_tuple is not None and enqueue_mask is not None:
            raise ValueError("indices_tuple and enqueue_mask are mutually exclusive")
        if enqueue_mask is not None:
            assert len(enqueue_mask) == len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        
        self.reset_stats()  # 重置统计信息
        
        # 设备一致性处理
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self.label_memory = c_f.to_device(
            self.label_memory, device=device, dtype=labels.dtype
        )
        self.timestamp_memory = c_f.to_device(  # 新增：时间戳设备同步
            self.timestamp_memory, device=device, dtype=torch.long
        )

        # 分离入队数据（保持原始逻辑）
        if enqueue_mask is not None:
            emb_for_queue = embeddings[enqueue_mask]
            labels_for_queue = labels[enqueue_mask]
            embeddings = embeddings[~enqueue_mask]
            labels = labels[~enqueue_mask]
            do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True

        # 更新内存（自动更新时间戳）
        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        # 计算时间衰减权重（新增）
        time_weights = self.compute_time_decay_weights(device)
        
        # 对内存嵌入应用时间衰减权重（关键修改）
        weighted_embedding_memory = self.embedding_memory * time_weights.unsqueeze(1)
        
        # 获取有效内存范围
        if not self.has_been_filled:
            E_mem = weighted_embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = weighted_embedding_memory
            L_mem = self.label_memory

        # 生成索引元组（保持原始逻辑）
        indices_tuple = self.create_indices_tuple(
            embeddings,
            labels,
            E_mem,
            L_mem,
            indices_tuple,
            do_remove_self_comparisons,
        )
        
        # 计算损失（使用加权后的内存嵌入）
        loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)
        return loss, indices_tuple

    def create_indices_tuple(self, embeddings, labels, E_mem, L_mem, 
                            input_indices_tuple, do_remove_self_comparisons):
        """（保持原始逻辑不变）"""
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)

        if do_remove_self_comparisons:
            indices_tuple = lmu.remove_self_comparisons(
                indices_tuple, self.curr_batch_idx, self.memory_size
            )

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = c_f.concatenate_indices_tuples(
                indices_tuple, input_indices_tuple
            )

        return indices_tuple