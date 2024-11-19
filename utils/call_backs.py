import pytorch_lightning.callbacks as plc
import sys
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


# define callback functions
class MyTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

    def init_validation_tqdm(self):
        bar = Tqdm(
            desc=self.validation_description,
            position=0,  # 这里固定写0
            disable=self.is_disabled,
            leave=True,  # leave写True
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


def load_callbacks(args):
    callbacks = []

    # use EarlyStopping.
    # The model will stop training after patience epoch where the monitor value (val_acc) is no longer increasing.
    # minimum change in the monitored quantity to qualify as an improvement,
    # i.e. an absolute change of less than or equal to `min_delta`, will count as no improvement.
    # todo: if we use multiple validation datasets, We must specify which dataset corresponds to the indicator being monitored.
    #  Same process should be setted in plc.ModelCheckpoint.
    if args.use_early_stopping:
        callbacks.append(plc.EarlyStopping(
            monitor=args.monitor_metric + '/R1',  # todo: change the monitor metric for your dataset
            mode='max',
            patience=args.patience,
            min_delta=0.00001
        ))

    #  the best k models according to the quantity monitored will be saved.
    callbacks.append(plc.ModelCheckpoint(
        # todo: change the monitor metric for your dataset
        monitor=args.monitor_metric + '/R1',
        # todo: change the monitor metric for your dataset
        filename=f'{args.model_name}' +
                 '_epoch({epoch:02d})_step({step:04d})_R1[{' + args.monitor_metric +
                 '/R1:.4f}]_R5[{' + args.monitor_metric +
                 '/R5:.4f}]_R10[{' + args.monitor_metric +
                 '/R10:.4f}]',
        save_top_k=3,
        auto_insert_metric_name=False,
        mode='max',
        save_last=True,
        save_weights_only=True
    ))

    # Generates a summary of all layers in a LightningModule
    # Note:The Trainer already configured with model summary callbacks by default.
    # callbacks.append(plc.ModelSummary(
    #     max_depth=1
    # ))

    # Automatically monitor and logs learning rate for learning rate schedulers during training.
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval='step'))

    # Change gradient accumulation factor according to scheduling.
    if args.gradient_accumulate:
        callbacks.append(plc.GradientAccumulationScheduler({args.gradient_accumulate_start_epoch:
                                                                args.gradient_accumulate_factor}))

    # Implements the Stochastic Weight Averaging (SWA) Callback to average a model.
    if args.StochasticWeightAveraging:
        callbacks.append(plc.StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.annealing_epochs,
            annealing_strategy=args.annealing_strategy,
        ))

    # init progress bar for validation
    callbacks.append(
        MyTQDMProgressBar()
    )

    return callbacks
