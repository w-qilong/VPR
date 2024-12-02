import importlib
import inspect

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
from utils import validation
from pytorch_metric_learning.losses import CrossBatchMemory
from losses import MetricLoss, LocalFeatureLoss


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
            print(f"Attempting to import module from: {module_path}")
            print(f"Looking for class: {camel_name}")

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
        if self.metric_loss_function in ["MultiSimilarityLoss", "TripletMarginLoss"]:
            self.metric_loss_function = MetricLoss(
                self.metric_loss_function, margin=self.hparams.miner_margin
            )
        else:
            raise ValueError(
                f'Optimizer {self.metric_loss_function} has not been added to "configure_loss()"'
            )

        self.local_loss_function = LocalFeatureLoss()

        # define memory bank
        if self.hparams.memory_bank:
            self.memory_bank = CrossBatchMemory(
                self.triplet_loss_function,
                self.hparams.mix_out_channels * self.hparams.mix_out_rows,
                memory_size=1024,
                miner=self.miner,
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

    def on_train_epoch_start(self):
        # 我们将跟踪损失层面上无效对/三元组的百分比
        self.triplet_batch_acc = []

    def training_step(self, batch, batch_idx):
        places, labels = batch

        # 注意GSVCities生成的places(每个包含N张图像)
        # 这意味着数据加载器将返回包含BS个places的batch
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        cls_token, fine_feature = self.forward(
            images
        )

        if (
            self.hparams.memory_bank
            and self.trainer.current_epoch > self.hparams.memory_bank_start_epoch
        ):
            metric_loss = self.memory_bank(cls_token, labels)
        else:
            # calculate metric loss
            metric_loss, miner_outputs = self.metric_loss_function(cls_token, labels)

        # 从miner_outputs中获取正负样本对的索引
        # a1: anchor样本的索引,用于与positive样本配对
        # p: positive样本的索引,与a1中的anchor样本配对形成正样本对
        # a2: anchor样本的索引,用于与negative样本配对
        # n: negative样本的索引,与a2中的anchor样本配对形成负样本对
        a1, p, a2, n = miner_outputs

        # 使用permute重新排列维度顺序
        a1_fused_features = fine_feature[a1]
        p_fused_features = fine_feature[p]
        a2_fused_features = fine_feature[a2]
        n_fused_features = fine_feature[n]

        simP = self.local_loss_function(a1_fused_features,p_fused_features).mean()
        simN = self.local_loss_function(a2_fused_features,n_fused_features).mean()
        # clamp函数用于将输入限制在指定范围内,这里设置min=0表示将所有小于0的值都设为0
        # 这样可以确保只有当负样本对的相似度大于正样本对的相似度时(即-simP+simN>0)才会产生损失
        total_local_loss = torch.sum(torch.clamp(-simP+simN+0., min=0.))
        
        # 计算总loss
        total_loss = metric_loss + total_local_loss

        # log metric loss and local loss
        self.log("metric_loss", metric_loss, prog_bar=True, logger=True)
        self.log("local_loss", total_local_loss, prog_bar=True, logger=True)

        # calculate the % of trivial pairs/triplets which do not contribute in the loss value
        nb_samples = cls_token.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        triplet_batch_acc = 1.0 - (nb_mined / nb_samples)

        # get mean accuracy
        self.triplet_batch_acc.append(triplet_batch_acc)
        self.log(
            "triplet_mean_acc",
            sum(self.triplet_batch_acc) / len(self.triplet_batch_acc),
            prog_bar=True,
            logger=True,
        )

        # return total loss
        return {"loss": total_loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.triplet_batch_acc = []

    def on_validation_epoch_start(self):
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.eval_set))]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # 计算描述符
        cls_token, fine_feature = self.forward(places)
        # 保存每个数据加载器的每个batch输出
        self.val_outputs[dataloader_idx].append(cls_token.detach().cpu())

    def on_validation_epoch_end(self):
        """返回按顺序排列的描述符
        取决于验证数据集的实现方式
        对于此项目(MSLS val, Pittburg val),始终是先参考图像后查询图像
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        k_values = self.hparams.recall_top_k  # recall K (1,5,10)
        val_step_outputs = self.val_outputs

        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.eval_dataset, dm.eval_set)
        ):
            if "pitts" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif "mapillary" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif "nordland" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif "spedtest" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif "essex3in1" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif "tokyo" in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()

            else:
                print(f"Please implement validation_epoch_end for {val_set_name}")
                raise NotImplemented

            # get and concat all global features
            cls_tokens = torch.concat(val_step_outputs[i], dim=0)
            ref_cls_tokens = cls_tokens[
                :num_references
            ]  # list of ref images descriptors
            query_cls_tokens = cls_tokens[
                num_references:
            ]  # list of query images descriptors
            # get the results of first ranking
            pitts_dict, predictions = validation.get_validation_recalls(
                r_list=ref_cls_tokens,
                q_list=query_cls_tokens,
                k_values=k_values,
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.hparams.faiss_gpu,
            )
            for k in k_values[:3]:
                self.log(
                    f"{val_set_name}/R{k}", pitts_dict[k], prog_bar=False, logger=True
                )

        # delete
        del (
            cls_tokens,
            ref_cls_tokens,
            query_cls_tokens,
            pitts_dict,
            predictions,
            val_step_outputs,
        )
        print("\n\n")

    def configure_optimizers(self):
        # 如果设置了weight_decay,将其值设置给优化器
        if hasattr(self.hparams, "weight_decay"):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        # set optimizer and its hparams
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
        if not self.hparams.lr_scheduler:
            return optimizer
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

            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]
