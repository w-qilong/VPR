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
        trainer.validate(
            model=model,
            datamodule=data_module,
            ckpt_path=args.ckpt_path,
            verbose=True,
        )
    else:
        # 正常训练流程
        trainer.fit(model, data_module)


if __name__ == "__main__":

    train = True
    # 获取checkpoint路径
    if not train:
        ckpt_path = 'logs/dinov2_backbone_dinov2_large/lightning_logs/version_26/checkpoints/dinov2_backbone_epoch(16)_step(16609)_R1[88.5100]_R5[94.1900]_R10[95.8100].ckpt'
    else:
        ckpt_path=None

    if ckpt_path:
        # 加载checkpoint对应的配置
        ckpt_config = load_checkpoint_config(ckpt_path)
        ckpt_config['ckpt_path'] = ckpt_path  # 设置默认值
        ckpt_config['eval_datasets'] = [
        # "mapillary_dataset",
        # 'spedtest_dataset',
        # 'tokyo247_dataset',
        # 'nordland_dataset',
        # 'pittsburg30k_dataset',

        # 'pittsburg250k_dataset',
        # 'gardenspoint_dataset',
        # 'stlucia_dataset',
        # 'eynsham_dataset',
        # 'svoxnight_dataset',
        # 'svoxrain_dataset',
        # 'amstertime_dataset',
        # 'essex3in1_dataset',
    ]
        ckpt_config['image_size_eval'] = [322, 322]
        args = argparse.Namespace(**ckpt_config)
    else:
        args = parser.parse_args()  

    main(args)
