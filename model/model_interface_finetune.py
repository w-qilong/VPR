import importlib
import inspect

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
from utils import losses, validation
from pytorch_metric_learning.losses import CrossBatchMemory


class AggMInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        # self.save_hyperparameters() Equivalent to self.hparams = hparams,
        # this line is equivalent to assigning a value to the self.hparams parameter
        self.kargs = kargs
        self.save_hyperparameters()

        self.load_model()
        self.configure_loss()
        self.save_hyperparameters()

    # load and init model by model file name and Class name.
    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')

        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
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
        # return global and ranked local features
        return self.model(x)

    def configure_loss(self):
        # define loss function
        self.triplet_loss_name = self.hparams.triplet_loss_function
        if self.triplet_loss_name in ['MultiSimilarityLoss', 'HardTripletLoss', 'TripletMarginLoss',
                                      'CentroidTripletLoss',
                                      'NTXentLoss', 'FastAPLoss', 'Lifted', 'ContrastiveLoss', 'CircleLoss',
                                      'SupConLoss']:
            self.triplet_loss_function = losses.get_loss(self.triplet_loss_name)
        else:
            raise ValueError(f'Optimizer {self.triplet_loss_name} has not been added to "configure_loss()"')

        # define triplet miner
        self.miner_name = self.hparams.miner_name
        if self.miner_name in ['TripletMarginMiner', 'MultiSimilarityMiner', 'PairMarginMiner']:
            self.miner = losses.get_miner(self.miner_name, self.hparams.miner_margin)
        else:
            raise ValueError(f'Optimizer {self.miner_name} has not been added to "configure_loss()"')

        # define memory bank
        if self.hparams.memory_bank:
            self.memory_bank = CrossBatchMemory(self.triplet_loss_function,
                                                self.hparams.mix_out_channels * self.hparams.mix_out_rows,
                                                memory_size=1024,
                                                miner=self.miner)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # manually warm up lr without a scheduler
        if self.hparams.warmup_steps and self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_start(self):
        # we will keep track of the % of trivial pairs/triplets at the loss level
        self.triplet_batch_acc = []

    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        global_feature = self.forward(images)

        # whether to use memory bank.
        # if we use memory bank, loss function should be set as memory bank
        # we use some epoch to init weights, then we use cross batch memory bank to mine hard/useful triplets
        if self.hparams.memory_bank and self.trainer.current_epoch > self.hparams.memory_bank_start_epoch:
            triplet_loss = self.memory_bank(global_feature, labels)
        else:
            # mining hard pairs which can contribute to loss
            miner_outputs = self.miner(global_feature, labels)
            # calculate loss
            triplet_loss = self.triplet_loss_function(global_feature, labels, miner_outputs)

        # miner_outputs = self.miner(global_feature, labels)
        # triplet_loss = self.triplet_loss_function(global_feature, labels, miner_outputs)
        self.log('loss', triplet_loss, prog_bar=True, logger=True)

        # calculate the % of trivial pairs/triplets
        # which do not contribute in the loss value
        # nb_samples = global_feature.shape[0]
        # nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        # triplet_batch_acc = 1.0 - (nb_mined / nb_samples)

        # get mean accuracy
        # self.triplet_batch_acc.append(triplet_batch_acc)
        # self.log('triplet_mean_acc', sum(self.triplet_batch_acc) / len(self.triplet_batch_acc), prog_bar=True,
        #          logger=True)

        # For MS loss, see https://blog.csdn.net/m0_46204224/article/details/117997854
        # MS-Loss包含两部分，前一部分是所有Positive Part对应的loss, 后分一部是所有Negative Part对应的loss。
        # anchor_index和positive_index记录Positive pair在global_feature的index。
        # 在此处，一个batch包含60个位置，一个位置包含四张图像。则anchor_index_positive, positive_index的长度为240*3，因为一张图像包含三张对应的positive图像。
        # anchor_index_negative, negative_index 的最大长度为240 *（240-4）=56640。但由于设置了对应的margin对负对进行筛选,
        # 所以anchor_index_negative, negative_index的长度小于56640

        return {'loss': triplet_loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.triplet_batch_acc = []

    def on_validation_epoch_start(self):
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.eval_set))]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # calculate descriptors
        global_feature = self.forward(places)
        # save each batch outputs for each dataloader
        self.val_outputs[dataloader_idx].append(global_feature.detach().cpu())

    def on_validation_epoch_end(self):
        """this return descriptors in their order
                depending on how the validation dataset is implemented
                for this project (MSLS val, Pittburg val), it is always references then queries
                [R1, R2, ..., Rn, Q1, Q2, ...]
                """
        dm = self.trainer.datamodule
        k_values = self.hparams.recall_top_k  # recall K (1,5,10)
        val_step_outputs = self.val_outputs

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.eval_dataset, dm.eval_set)):
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'mapillary' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'nordland' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'spedtest' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'essex3in1' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'tokyo' in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()

            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            # get and concat all global features
            global_feats = torch.concat(val_step_outputs[i], dim=0)
            ref_global_list = global_feats[: num_references]  # list of ref images descriptors
            query_global_list = global_feats[num_references:]  # list of query images descriptors
            # get the results of first ranking
            pitts_dict, predictions = validation.get_validation_recalls(r_list=ref_global_list,
                                                                        q_list=query_global_list,
                                                                        k_values=k_values,
                                                                        gt=positives,
                                                                        print_results=True,
                                                                        dataset_name=val_set_name,
                                                                        faiss_gpu=self.hparams.faiss_gpu
                                                                        )
            for k in k_values[:3]:
                self.log(f'{val_set_name}/R{k}', pitts_dict[k], prog_bar=False, logger=True)

        # delete
        del global_feats, ref_global_list, query_global_list, pitts_dict, predictions, val_step_outputs
        print('\n\n')

    def configure_optimizers(self):
        # If weight_decay is set, set its value to optimizer
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        # set optimizer and its hparams
        if self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        weight_decay=weight_decay,
                                        momentum=self.hparams.momentum)

        elif self.hparams.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=weight_decay)

        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.hparams.lr,
                                         weight_decay=weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

        # Use lr_scheduler
        if not self.hparams.lr_scheduler:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)

            elif self.hparams.lr_scheduler == 'multi_step':
                scheduler = lrs.MultiStepLR(optimizer,
                                            milestones=self.hparams.milestones,
                                            gamma=self.hparams.lr_decay_rate)

            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.T_max,
                                                  eta_min=self.hparams.eta_min)

            elif self.hparams.lr_scheduler == 'linear':
                scheduler = lrs.LinearLR(
                    optimizer,
                    start_factor=self.hparams.start_factor,
                    end_factor=self.hparams.end_factor,
                    total_iters=self.hparams.total_iters
                )

            elif self.hparams.lr_scheduler == 'exp':
                scheduler = lrs.ExponentialLR(
                    optimizer,
                    gamma=self.hparams.gamma
                )

            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
