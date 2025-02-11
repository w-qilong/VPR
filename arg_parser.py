from argparse import ArgumentParser

parser = ArgumentParser()

# todo: Model Hyperparameters
parser.add_argument("--model_name", default="dinov2_backbone", type=str)
parser.add_argument("--backbone_size", default="dinov2_large", type=str)
parser.add_argument("--lora_r", default=8, type=int)
parser.add_argument("--lora_alpha", default=16, type=int)
parser.add_argument("--lora_dropout", default=0.1, type=float)
parser.add_argument("--finetune_last_n_layers", default=6, type=int)
parser.add_argument("--reduced_dim", default=1024, type=int)

# todo:rerank
parser.add_argument("--rerank", default=False, type=bool)
parser.add_argument("--saliency_thresh", default=0.55, type=float)
parser.add_argument("--nn_match_thresh", default=0.8, type=float)
parser.add_argument(
    "--facet_layer_and_facet", default={22: "value", 23: "attn"}, type=dict
)
parser.add_argument("--include_cls", default=False, type=bool)
parser.add_argument("--bin", default=False, type=bool)
parser.add_argument("--hierarchy", default=2, type=int)

# todo: Datasets information
# Typically, we need to verify the performance of our model on multiple validation datasets.
# Here, we can assign train/eval/test datasets. Here, we use standard_data for
parser.add_argument("--train_dataset", default="gsvcities_dataset", type=str)
# args for training dataset GSVCities
parser.add_argument("--image_size_train", default=[224, 224], type=list)
parser.add_argument("--image_size_eval", default=[322, 322], type=list)
parser.add_argument("--shuffle_all", default=True, type=bool)
parser.add_argument("--img_per_place", default=4, type=int)
parser.add_argument("--min_img_per_place", default=4, type=int)
parser.add_argument("--random_sample_from_each_place", default=True, type=bool)
parser.add_argument("--persistent_workers", default=False, type=bool)
# args for eval dataset
parser.add_argument(
    "--eval_datasets",
    default=[
        "mapillary_dataset",
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
    ],
    type=list,
)

# set monitor dataset
parser.add_argument("--monitor_metric", default="mapillary_dataset", type=str)
parser.add_argument("--recall_top_k", default=[1, 5, 10], type=list)

# todo: Basic Training Control for global trainer
# set random seed
parser.add_argument("--seed", default=1234, type=int)
# use GPU or CPU
parser.add_argument("--accelerator", default="gpu", type=str)
# select GPU device
parser.add_argument("--devices", default=[0], type=list)
# set training epochs
parser.add_argument("--epochs", default=30, type=int)
# set batch size
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=64, type=int)

# set number of process worker in dataloader
parser.add_argument("--num_workers", default=15, type=int)
# set init learning rate for global trainer
parser.add_argument("--lr", default=1e-5, type=float)
# select optimizer. We have defined multiple optimizers in model_interface.py, we can select one for our study here.
parser.add_argument("--optimizer", choices=["sgd", "adamw", "adam"], default="adam", type=str)
# set momentum of optimizer. It should set for sgd. When we use adam or adamw optimizer, no need to set it
parser.add_argument("--momentum", default=0.9, type=float)
# set weight_decay rate for optimizer
parser.add_argument("--weight_decay", default=9.5e-09, type=float)

# todo: LR Scheduler. Used for dynamically adjusting learning rates
# Here, we can use gradual warmup to , i.e., start with an initially small learning rate,
# and increase a little bit for each STEP until the initially set relatively large learning rate is reached,
# and then use the initially set learning rate for training.
parser.add_argument("--warmup_steps", default=200, type=int)

# select lr_scheduler. We have defined multiple lr_scheduler in model_interface.py, we can select one for our study here.
parser.add_argument(
    "--lr_scheduler",
    choices=["none", "step", "multi_step", "cosine", "linear", "exp"],
    default="none",
    type=str,
)

# Set args for Different Scheduler
# For CosineAnnealingLR
# parser.add_argument("--T_max", default=5, type=int)
# parser.add_argument("--eta_min", default=1e-6, type=float)

# For StepLR
# parser.add_argument('--lr_decay_steps', default=3, type=int)
# parser.add_argument('--lr_decay_rate', default=0.5, type=float)

# For MultiStepLR
# parser.add_argument('--milestones', default=[30, 35, 40], type=list)
# # lr_decay_rate controls the change rate of learning rate
# parser.add_argument('--lr_decay_rate', default=0.5, type=float)

# For LinearLR
# parser.add_argument('--start_factor', default=1, type=float)
# parser.add_argument('--end_factor', default=0.2, type=float)
# parser.add_argument('--total_iters', default=1000 * 100, type=int)

# For ExponentialLR
# parser.add_argument("--gamma", default=0.99, type=float)


# todo: loss function
# select loss function. We have defined multiple loss function in model_interface.py,
# set args for loss function and triplet miner
parser.add_argument(
    "--metric_loss_function",
    choices=["MultiSimilarityLoss", "TripletMarginLoss", "ContrastiveLoss", "NCALoss"],
    default="MultiSimilarityLoss",
    type=str,
)

parser.add_argument(
    "--miner_name",
    choices=["MultiSimilarityMiner", "TripletMarginMiner", "PairMarginMiner"],
    default="MultiSimilarityMiner",
    type=str,
)

# set margin for miner
parser.add_argument("--miner_margin", default=0.1, type=float)

# whether to use memory bank
parser.add_argument("--memory_bank", default=False, type=bool)
parser.add_argument("--memory_bank_start_epoch", default=5, type=int)
parser.add_argument("--memory_bank_size", default=2048, type=int)  # 4*64*4= 1024 4*64*8=2048 4*64*16=4096
parser.add_argument("--decay_lambda", default=0, type=float)
parser.add_argument("--save_feats", default=False, type=bool)
parser.add_argument("--save_neg_num", default=False, type=bool)

# whether to use gpu for calculate distance for validation
parser.add_argument("--faiss_gpu", default=False, type=bool)

# whether use gradient accumulate
parser.add_argument("--gradient_accumulate", default=False, type=bool)
parser.add_argument("--gradient_accumulate_start_epoch", default=0, type=int)
parser.add_argument("--gradient_accumulate_factor", default=2, type=int)

# whether to use early stopping
parser.add_argument("--use_early_stopping", default=False, type=bool)
parser.add_argument("--patience", default=5, type=int)

# set if StochasticWeightAveraging need to be used
parser.add_argument("--StochasticWeightAveraging", default=False, type=bool)
parser.add_argument("--swa_lrs", default=1e-5, type=float)
parser.add_argument("--swa_epoch_start", default=0.75, type=float)
parser.add_argument("--annealing_epochs", default=10, type=int)
parser.add_argument("--annealing_strategy", default="cos", type=str)
