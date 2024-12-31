import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from prettytable import PrettyTable


# Here, we define the interface inherit from pl.LightningDataModule.
# We can control the batch size, train/eval/test datasets used in our study by args from main.py,
# because all args can be input to DInterface by **kwargs in __init__ function.
class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._init_config(kwargs)
        self._init_transforms()
        self._init_dataloader_config()
        self.save_hyperparameters()

    def _init_config(self, kwargs):
        # 将配置参数初始化集中到一个方法中
        self.train_dataset = kwargs["train_dataset"]
        self.eval_dataset = kwargs["eval_datasets"]
        self.num_workers = kwargs["num_workers"]
        self.batch_size = kwargs["batch_size"]
        self.image_size_train = kwargs["image_size_train"]
        self.image_size_eval = kwargs["image_size_eval"]
        self.shuffle_all = kwargs["shuffle_all"]
        self.img_per_place = kwargs["img_per_place"]
        self.min_img_per_place = kwargs["min_img_per_place"]
        self.random_sample_from_each_place = kwargs["random_sample_from_each_place"]
        self.persistent_workers = kwargs["persistent_workers"]
        
    def _init_transforms(self):
        # 将transforms初始化集中到一个方法中
        mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        
        self.train_transform = T.Compose([
            T.Resize(tuple(self.image_size_train), interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(**mean_std),
        ])
        
        self.valid_transform = T.Compose([
            T.Resize(tuple(self.image_size_eval), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(**mean_std),
        ])

    def _init_dataloader_config(self):
        # 将dataloader配置集中到一个方法中
        base_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "persistent_workers": self.persistent_workers,
        }
        
        self.train_loader_config = {**base_config, "shuffle": self.shuffle_all}
        self.valid_loader_config = {**base_config, "shuffle": False} # 不打乱数据集顺序，便于重排序

    def load_data_module(self, dataset_name):
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in dataset_name.split("_")])
        try:
            data_module = getattr(
                importlib.import_module("." + dataset_name, package=__package__),
                camel_name,
            )
            return data_module
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{dataset_name}.{camel_name}"
            )

    def instancialize(self, data_module, **other_args):
        """Instancialize a data Class using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return data_module(**args1)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # init train dataset
            if self.train_dataset == "gsvcities_dataset":
                self.train_set = self.instancialize(
                    self.load_data_module(self.train_dataset),
                    img_per_place=self.img_per_place,
                    min_img_per_place=self.min_img_per_place,
                    random_sample_from_each_place=self.random_sample_from_each_place,
                    transform=self.train_transform,
                )
            # init eval datasets
            self.eval_set = []
            for item in self.eval_dataset:
                self.eval_set.append(
                    self.instancialize(
                        self.load_data_module(item),
                        input_transform=self.valid_transform,
                    )
                )

            # show dataset information
            self.print_stats()

        if stage == "validate":
            # init eval datasets
            self.eval_set = []
            for item in self.eval_dataset:
                self.eval_set.append(
                    self.instancialize(
                        self.load_data_module(item),
                        input_transform=self.valid_transform,
                    )
                )

        # Assign test dataset for use in dataloader(s)
        # you can put multiple dataset into a list and return list for test
        if stage == "test":
            # init test datasets
            self.test_set = []
            for item in self.test_dataset:
                self.test_set.append(self.instancialize(self.load_data_module(item)))

    def train_dataloader(self):
        dataset_cls = self.load_data_module(self.train_dataset)
        self.train_set = dataset_cls(
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            **self.kwargs
        )
        return DataLoader(self.train_set, **self.train_loader_config)

    def val_dataloader(self):
        eval_dataloaders = []
        for dataset in self.eval_set:
            eval_dataloaders.append(
                DataLoader(dataset=dataset, **self.valid_loader_config)
            )
        return eval_dataloaders

    def test_dataloader(self):
        test_dataloaders = []
        for dataset in self.test_set:
            test_dataloaders.append(
                DataLoader(dataset=dataset, **self.valid_loader_config)
            )
        return test_dataloaders

    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        table.add_row(["# of cities", f"{len(self.train_set.cities)}"])
        table.add_row(["# of places", f"{self.train_set.__len__()}"])
        table.add_row(["# of images", f"{self.train_set.total_nb_images}"])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.eval_dataset):
            table.add_row([f"Validation set {i + 1}", f"{val_set_name}"])
        # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_set.__len__() // self.batch_size}"]
        )
        table.add_row(["Image size", f"{self.image_size_train}"])
        print(table.get_string(title="Training config"))
