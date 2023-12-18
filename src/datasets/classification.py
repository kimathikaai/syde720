import os

import lightning as L
import numpy as np
from cv2 import transform
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


class ClassDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        img_h: int = 224,
        img_w: int = 224,
        img_c=3,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None) -> None:
        #
        # Transforms
        #
        self.transform = T.Compose(
            [
                T.RandomResizedCrop((self.img_h, self.img_w), scale=(0.75, 1.0)),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ]
        )

        #
        # Datasets
        #
        self.train_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "training"), transform=self.transform
        )
        self.val_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "validation"), transform=self.transform
        )
        self.test_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "evaluation"), transform=self.transform
        )
        print("Created classification datasets")

        # Update the number of classes
        self.classes = np.unique(self.train_dataset.targets)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    data_module = ClassDataModule(
        data_dir="/Users/kimathikaai/workspace/syde720/tmp/food-11",
        batch_size=8,
        num_workers=0,
    )
    data_module.setup()

    print(data_module.train_dataset)
    print(data_module.val_dataset)
    print(data_module.test_dataset)
    import pdb

    pdb.set_trace()
