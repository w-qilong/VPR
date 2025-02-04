""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""

import warnings
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
import numpy as np

from data import DInterface
from model import AggMInterface
from arg_parser import parser
from utils.load_config import load_checkpoint_config


# import call callbacks functions and parser for args
from utils.call_backs import load_callbacks

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("high")


def main(args):
    # set random seed
    pl.seed_everything(args.seed)

    # init pytorch_lighting data and model module
    # vars(args) transformer property and value of a python object into a dict
    data_module = DInterface(**vars(args))
    model = AggMInterface(**vars(args))

    # add callbacks to args and send it to Trainer
    args.callbacks = load_callbacks(args)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=f"./logs/{args.model_name}_{args.backbone_size}",
        # we use current model for log folder name
        max_epochs=args.epochs,
        callbacks=args.callbacks,  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        check_val_every_n_epoch=1,  # run validation every epoch
        log_every_n_steps=5,
        enable_model_summary=True,
        benchmark=True,
        num_sanity_val_steps=0,  # runs a validation step before starting training
        precision="16-mixed",  # we use half precision to reduce  memory usage
        accumulate_grad_batches=(
            args.gradient_accumulate_factor if args.gradient_accumulate else 1
        ),
        # todo: used for debug
        # profiler=profiler,
        # fast_dev_run=True,  # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
        # limit_train_batches=1,
        # limit_val_batches=2
    )

      # 如果指定了checkpoint，进行测试
    if hasattr(args, 'ckpt_path'):
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)
        trainer.validate(
            model=model,
            datamodule=data_module,
            # ckpt_path=args.ckpt_path,
            verbose=True,
        )
    else:
        # 正常训练流程
        trainer.fit(model, data_module)


if __name__ == "__main__":

    # single train
    # args = parser.parse_args()
    # main(args)


    # # 测试不同memory_bank_start_epoch对结果的影响
    # for i in [True,False]:
    #     # 获取checkpoint路径
    #     args = parser.parse_args()  
    #     args.memory_bank = i
    #     main(args)


    checkpoint_paths=[
        'logs/dinov2_backbone_dinov2_large/lightning_logs/version_14/checkpoints/dinov2_backbone_epoch(39)_step(39080)_R1[0.9135]_R5[0.9595]_R10[0.9649].ckpt',
    ]

    # 测试不同checkpoint对结果的影响
    # 加载checkpoint对应的配置
    for ckpt_path in checkpoint_paths:
        ckpt_config = load_checkpoint_config(ckpt_path)
        ckpt_config['ckpt_path'] = ckpt_path  # 设置默认值
        ckpt_config['eval_datasets'] = [
        # "mapillary_dataset",
        # 'tokyo247_dataset',
        # 'nordland_dataset',
        'pittsburg30k_dataset',
        'spedtest_dataset',

        'stlucia_dataset',
        'eynsham_dataset',
        'svoxnight_dataset',
        'svoxrain_dataset',
        'amstertime_dataset',

        # 'essex3in1_dataset',
        # 'pittsburg250k_dataset',
        # 'gardenspoint_dataset',
    ]
        ckpt_config['image_size_eval'] = [560, 560]
        ckpt_config['rerank'] = True
        ckpt_config['facet_layer_and_facet'] = {22: "value", 23: "attn"}

        args = argparse.Namespace(**ckpt_config)

        # 执行main函数
        main(args)
