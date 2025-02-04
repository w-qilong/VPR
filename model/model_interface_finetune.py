import importlib
import inspect
import numpy as np
from typing import Dict, Any
from prettytable import PrettyTable
import pytorch_lightning as pl
import torch
from tqdm import tqdm
import torch.optim.lr_scheduler as lrs
from utils import validation
from pytorch_metric_learning.losses import CrossBatchMemory
from losses import MetricLoss
from utils import hook_func
from PIL import Image
import os
import pandas as pd
import time


class AggMInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        # self.save_hyperparameters() 等同于 self.hparams = hparams,
        # 这行代码相当于给self.hparams参数赋值
        self.kargs = kargs
        self.save_hyperparameters()

        self.load_model()
        self.configure_loss()
        self.save_hyperparameters()

    # 通过模型文件名和类名加载并初始化模型
    def load_model(self):
        name = self.hparams.model_name
        # 将`snake_case.py`文件名转换为`CamelCase`类名
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            # 添加调试信息
            module_path = "." + name
            print(f"正在尝试从以下路径导入模块: {module_path}")
            print(f"正在查找类: {camel_name}")

            module = importlib.import_module("." + name, package=__package__)
            Model = getattr(module, camel_name)
        except ImportError as e:
            raise ValueError(
                f"Could not import module {name}. Error: {str(e)}\n"
                f"Please check if the file {name}.py exists in the correct directory."
            )
        except AttributeError as e:
            raise ValueError(
                f"Could not find class {camel_name} in module {name}.\n"
                f"Please check if the class name matches the file name (converted to CamelCase)."
            )
        except Exception as e:
            raise ValueError(
                f"Error loading model: {str(e)}\n"
                f"Module: {name}, Expected class: {camel_name}"
            )

        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """使用self.hparams字典中的相应参数实例化模型。
        你也可以输入任何参数来覆盖self.hparams中的对应值。
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def forward(self, x):
        # 前向传播
        return self.model(x)

    def configure_loss(self):
        # 定义损失函数
        self.metric_loss_function = self.hparams.metric_loss_function
        if self.metric_loss_function in ["MultiSimilarityLoss", "TripletMarginLoss", "ContrastiveLoss", "NCALoss"]:
            self.metric_loss_function = MetricLoss(
                self.metric_loss_function, margin=self.hparams.miner_margin
            )
        else:
            raise ValueError(
                f'Optimizer {self.metric_loss_function} has not been added to "configure_loss()"'
            )

        # define memory bank
        if self.hparams.memory_bank:
            self.memory_bank = CrossBatchMemory(
                self.metric_loss_function.loss_fn,
                self.model.num_features,
                memory_size=self.hparams.memory_bank_size,
                miner=self.metric_loss_function.miner,
                decay_lambda=self.hparams.decay_lambda,
            )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # 手动预热学习率,不使用调度器
        if (
            self.hparams.warmup_steps
            and self.trainer.global_step < self.hparams.warmup_steps
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_train_start(self):

        if self.hparams.save_neg_num:
            # 初始化列表存储负样本数量
            self.neg_num_list = []

        # 初始化图像
        if self.hparams.save_feats:
            # 初始化列表存储特征
            self.feats_list = []
            # 实例化两个pair图像，用于记录模型训练过程中，特征提取器提取的特征
            self.img_paths=[
                'tmp_imgs/00010.jpg',
                'tmp_imgs/00012.jpg',
                'tmp_imgs/0000046.jpg',
                'tmp_imgs/0000047.jpg'
            ]
            self.tmp_imgs = [Image.open(img_path).convert('RGB') for img_path in self.img_paths]
            transform = self.trainer.datamodule.valid_transform
            self.tmp_imgs = torch.stack([transform(img) for img in self.tmp_imgs])

    def on_train_end(self):
        # 序列化
        if self.hparams.save_feats:
            save_path=os.path.join(self.trainer.log_dir, 'feats_list.pth')
            torch.save(self.feats_list, save_path)

        if self.hparams.save_neg_num:    
            save_path=os.path.join(self.trainer.log_dir, 'neg_num_list.pth')
            torch.save(self.neg_num_list, save_path)

    def on_train_epoch_start(self):
        # 我们将跟踪损失层面上无效对/三元组的百分比
        self.triplet_batch_acc = []
            
    def training_step(self, batch, batch_idx):       
        # 获取batch数据
        places, labels = batch

        # 注意GSVCities生成的places(每个包含N张图像)
        # 这意味着数据加载器将返回包含BS个places的batch
        BS, N, ch, h, w = places.shape

        # 重塑places和labels的维度
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # 将batch输入模型进行前向传播
        cls_token = self.forward(images)

        if (
            self.hparams.memory_bank
            and self.trainer.current_epoch >= self.hparams.memory_bank_start_epoch
        ):
            # 使用memory bank
            metric_loss, miner_outputs = self.memory_bank(cls_token, labels)
            
            # 记录负样本数量
            if self.hparams.save_neg_num:
                # 使用memory bank
                a1, p, a2, n2 = miner_outputs
                # 不使用memory bank
                with torch.no_grad():
                    _, miner_outputs = self.metric_loss_function(cls_token, labels)
                    a1, p, a2, n1 = miner_outputs
                self.neg_num_list.append([len(n1), len(n2)])

        else:
            # 计算度量损失
            metric_loss, miner_outputs = self.metric_loss_function(cls_token, labels)

            # calculate the % of trivial pairs/triplets which do not contribute in the loss value
            nb_samples = cls_token.shape[0]

            # 当使用NCALoss时，没有miner
            if self.metric_loss_function.miner:
                nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
                triplet_batch_acc = 1.0 - (nb_mined / nb_samples)
            else:
                triplet_batch_acc = 0.0

            # get mean accuracy
            self.triplet_batch_acc.append(triplet_batch_acc)
            self.log(
                "mean_acc",
                sum(self.triplet_batch_acc) / len(self.triplet_batch_acc),
                prog_bar=True,
                logger=True,
            )

        # 从miner_outputs中获取正负样本对的索引
        # a1: anchor样本的索引,用于与positive样本配对
        # p: positive样本的索引,与a1中的anchor样本配对形成正样本对
        # a2: anchor样本的索引,用于与negative样本配对
        # n: negative样本的索引,与a2中的anchor样本配对形成负样本对
        # a1, p, a2, n = miner_outputs

        # log metric loss and local loss
        self.log("metric_loss", metric_loss, prog_bar=True, logger=True)

        # 保存特征
        if self.hparams.save_feats:
            self.predict_step(batch, batch_idx)

        # return total loss
        return {"loss": metric_loss}
    
    def predict_step(self, batch, batch_idx):
        # 获取batch数据
        with torch.no_grad():
            batch_feats = self.forward(self.tmp_imgs.cuda())
            self.feats_list.append(batch_feats.detach().cpu())

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.triplet_batch_acc = []
        if self.hparams.memory_bank:
            self.memory_bank.reset_queue()

    def on_validation_epoch_start(self):
        self.val_cls_outputs = [
            [] for _ in range(len(self.trainer.datamodule.eval_set))
        ]

        # 初始化列表，用于存储排序结果
        self.val_results = dict()                   

        if self.hparams.rerank:
            # 初始化列表，用于存储重排序结果
            self.val_rerank_results = dict()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            places, _ = batch
            # 简化为只获取cls_token
            cls_token = self.forward(places)
            self.val_cls_outputs[dataloader_idx].append(cls_token.detach().cpu())

    def get_dataset_info(self, val_set_name, val_dataset):
        """获取数据集的基本信息"""
        dataset_configs = {
            "pitts": lambda: (val_dataset.dbStruct.numDb, val_dataset.getPositives()),
            "mapillary": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "nordland": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "spedtest": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "essex3in1": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "tokyo": lambda: (val_dataset.dbStruct.numDb, val_dataset.getPositives()),
            "gardenspoint": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "stlucia": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "eynsham": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "svoxnight": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "svoxrain": lambda: (val_dataset.num_references, val_dataset.pIdx),
            "amstertime": lambda: (val_dataset.num_references, val_dataset.pIdx),
        }

        for key in dataset_configs:
            if key in val_set_name:
                return dataset_configs[key]()

        raise NotImplementedError(f"请实现{val_set_name}的validation_epoch_end")

    def on_validation_epoch_end(self):
        dm = self.trainer.datamodule
        k_values = self.hparams.recall_top_k

        # 遍历每个验证集
        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.eval_dataset, dm.eval_set)
        ):
            # 获取数据集信息
            num_references, positives = self.get_dataset_info(val_set_name, val_dataset)

            # 获取特征并分割
            cls_tokens = torch.concat(self.val_cls_outputs[i], dim=0)
            ref_cls_tokens = cls_tokens[:num_references]
            query_cls_tokens = cls_tokens[num_references:]

            # 第一次排序
            pitts_dict, first_predictions = validation.get_validation_recalls(
                r_list=ref_cls_tokens,
                q_list=query_cls_tokens,
                k_values=k_values,
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.hparams.faiss_gpu,
            )

            # 记录第一次结果
            for k in k_values:
                self.log(
                    f"{val_set_name}/R{k}", pitts_dict[k], prog_bar=False, logger=True
                )

            # 保存第一次结果
            self.val_results[val_set_name] = pitts_dict

            if self.hparams.rerank:
                # 第二次排序
                saliency_threshs = np.arange(0.1, 0.9, 0.1).tolist()
                nn_match_threshs = np.arange(0.1, 0.9, 0.1).tolist()
                
                for saliency_thresh in saliency_threshs:
                    for nn_match_thresh in nn_match_threshs:

                        print(f"saliency_thresh: {np.round(saliency_thresh, 2)}, nn_match_thresh: {np.round(nn_match_thresh, 2)}")
                        print(f"\n开始对{val_set_name}数据集进行二次排序...")

                        rerank_predictions = self.rerank_predictions(
                            val_dataset, num_references, first_predictions,
                            saliency_thresh=saliency_thresh,
                            nn_match_thresh=nn_match_thresh,
                        )

                        # 计算并记录第二次结果
                        d, rerank_predictions = self.calculate_rerank_metrics(
                            rerank_predictions, positives, k_values, val_set_name
                        )

                        # 记录第二次结果
                        for k in k_values:
                            self.log(
                                f"{val_set_name}_{saliency_thresh}_{nn_match_thresh}_rerank/R{k}", d[k], prog_bar=False, logger=True
                            )

                        # 保存第二次结果
                        key = f"{val_set_name}_{saliency_thresh}_{nn_match_thresh}"
                        self.val_rerank_results[key] = d

                        del d, rerank_predictions

                        if self.hparams.rerank:
                            # 保存第二次结果
                            save_path=os.path.join(self.trainer.log_dir, 'second_predictions.xlsx')
                            df= pd.DataFrame.from_dict(self.val_rerank_results, orient='index')
                            df.to_excel(save_path)

            # 清理内存
            del (
                cls_tokens,
                ref_cls_tokens,
                query_cls_tokens,
                pitts_dict,
            )
            torch.cuda.empty_cache()

            print("\n")

        # 保存第一次结果
        save_path=os.path.join(self.trainer.log_dir, 'first_predictions.xlsx')
        df= pd.DataFrame.from_dict(self.val_results, orient='index')
        df.to_excel(save_path)

        if self.hparams.rerank:
            # 保存第二次结果
            save_path=os.path.join(self.trainer.log_dir, 'second_predictions.xlsx')
            df= pd.DataFrame.from_dict(self.val_rerank_results, orient='index')
            df.to_excel(save_path)

    def calculate_rerank_metrics(
        self, rerank_predictions, positives, k_values, val_set_name
    ):
        """计算重排序的recall@k"""
        rerank_predictions = np.array(rerank_predictions)
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(rerank_predictions):
            for i, n in enumerate(k_values):
                if np.any(np.in1d(pred[:n], positives[q_idx])):
                    correct_at_k[i:] += 1
                    break

        correct_at_k = correct_at_k / len(rerank_predictions)
        d = {k: np.round(v*100, 2) for (k, v) in zip(k_values, correct_at_k)}

        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["K"] + [str(k) for k in k_values]
        table.add_row(["Recall@K"] + [f"{100 * v:.2f}" for v in correct_at_k])
        print(table.get_string(title=f"Performances rerank on {val_set_name}"))

        return d, rerank_predictions

    def rerank_predictions(self, val_dataset, num_references, first_predictions,saliency_thresh,nn_match_thresh):
        """执行重排序"""
        rerank_predictions = []

        with torch.no_grad():
            for query_idx in tqdm(range(len(first_predictions))):
                # 获取查询和参考图像
                query_image = (
                    val_dataset[query_idx + num_references][0]
                    .unsqueeze(0)
                    .to(self.device)
                )
                candidate_ref_indices = first_predictions[query_idx]
                ref_images = torch.stack(
                    [val_dataset[i][0] for i in candidate_ref_indices]
                ).to(self.device)

                # 获取特征和注意力图
                query_saliency_map, query_feats = self.get_features_and_attention(
                    query_image
                )
                ref_saliency_map, ref_feats = self.get_features_and_attention(
                    ref_images
                )
                # 重排序单个查询
                rerank_predictions.append(
                    validation.single_rerank(
                        query_feats,
                        ref_feats,
                        query_saliency_map,
                        ref_saliency_map,
                        candidate_ref_indices,
                        saliency_thresh=saliency_thresh,
                        nn_match_thresh=nn_match_thresh,
                    )
                )

        return rerank_predictions

    def get_features_and_attention(self, images):
        """获取特征和注意力图"""
        feats = []
        hook_handlers = []

        # 注册hooks
        facet_layer_and_facet = self.hparams.facet_layer_and_facet
        hook_func.register_hooks(
            self.model.model, facet_layer_and_facet, hook_handlers, feats
        )

        # 前向传播
        self.forward(images)

        # 取消hooks
        hook_func.unregister_hooks(hook_handlers)

        return self.process_attention_and_features(
            (feats[0], feats[1]),
            facet_layer_and_facet,
            self.hparams.include_cls,
            self.hparams.bin,
            self.hparams.hierarchy,
        )

    def process_attention_and_features(
        self,
        feats_and_saliency_map=(None, None),
        facet_layer_and_facet={22: "value", 23: "attn"},
        include_cls=False,
        bin=False,
        hierarchy=2,
    ):
        """处理注意力图和特征

        Args:
            feats_and_saliency_map (tuple): 包含两个元素:
                - feats_and_saliency_map[0]: token特征 [B, h, t, d] 或 [B, t, d]
                - feats_and_saliency_map[1]: 注意力特征 [B, h, t, t]
            facet_layer_and_facet (dict): 包含不同层的facet信息
            include_cls (bool): 是否包含CLS token
            bin (bool): 是否使用log bin处理
            hierarchy (int): log bin的层级数

        Returns:
            tuple: (saliency_map, desc)
                - saliency_map: 显著性图 [B, t-1]
                - desc: 描述符 [B, 1, t, d*h] 或 [B, 1, t-1, d*h*num_bins]
        """
        # 1. 处理注意力图
        saliency_map = feats_and_saliency_map[1]  # [B, h, t, t]

        # 提取CLS token的注意力并计算平均值
        saliency_map = saliency_map[:, :, 0, 1:].mean(dim=1)  # [B, t-1]

        # 对每个样本进行归一化
        temp_mins = saliency_map.min(dim=1)[0]  # [B]
        temp_maxs = saliency_map.max(dim=1)[0]  # [B]
        saliency_map = (saliency_map - temp_mins.unsqueeze(1)) / (
            temp_maxs - temp_mins
        ).unsqueeze(
            1
        )  # [B, t-1]

        # 2. 处理特征
        tokens_num = feats_and_saliency_map[1].shape[2]
        num_patches = (int(tokens_num**0.5), int(tokens_num**0.5))
        facet = list(facet_layer_and_facet.values())[0]

        feats = feats_and_saliency_map[0]  # [B, h, t, d] 或 [B, t, d]

        # 统一token特征的维度格式
        if facet == "token":
            feats = feats.unsqueeze(1)  # [B, 1, t, d]

        # 处理CLS token
        if not include_cls:
            feats = feats[:, :, 1:]  # [B, h, t-1, d]

        # 生成描述符
        if not bin:
            feats = feats.permute(0, 2, 3, 1)  # [B, t-1, d, h]
            feats = feats.flatten(start_dim=-2)  # [B, t-1, d*h]
            feats = feats.unsqueeze(1)  # [B, 1, t-1, d*h]
        else:
            feats = hook_func.log_bin(
                feats, num_patches, hierarchy=hierarchy
            )  # [B, 1, t-1, d*h*num_bins]

        return saliency_map, feats

    def configure_optimizers(self):
        # 如果设置了weight_decay,将其值设置给优化器
        if hasattr(self.hparams, "weight_decay"):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        # 设置优化器及其超参数
        if self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=weight_decay,
                momentum=self.hparams.momentum,
            )

        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )

        elif self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"'
            )

        # Use lr_scheduler
        if self.hparams.lr_scheduler == "none":
            return [optimizer]  # 不使用调度器
        else:
            if self.hparams.lr_scheduler == "step":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )

            elif self.hparams.lr_scheduler == "multi_step":
                scheduler = lrs.MultiStepLR(
                    optimizer,
                    milestones=self.hparams.milestones,
                    gamma=self.hparams.lr_decay_rate,
                )

            elif self.hparams.lr_scheduler == "cosine":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.eta_min
                )

            elif self.hparams.lr_scheduler == "linear":
                scheduler = lrs.LinearLR(
                    optimizer,
                    start_factor=self.hparams.start_factor,
                    end_factor=self.hparams.end_factor,
                    total_iters=self.hparams.total_iters,
                )

            elif self.hparams.lr_scheduler == "exp":
                scheduler = lrs.ExponentialLR(optimizer, gamma=self.hparams.gamma)

            elif self.hparams.lr_scheduler == "cosine_with_restarts":
                scheduler = lrs.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.hparams.T_0, T_mult=self.hparams.T_mult
                )

            elif self.hparams.lr_scheduler == "plateau":
                scheduler = lrs.ReduceLROnPlateau(
                    optimizer, mode="min", factor=self.hparams.lr_decay_rate, patience=10
                )

            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]
        
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 过滤不需要的键
        state_dict = checkpoint["state_dict"]
        self.load_state_dict(state_dict, strict=False)
        