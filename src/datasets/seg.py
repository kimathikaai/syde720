import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.target_transform = transforms.Compose([transforms.ToTensor()])

        # self.dims = (3, 32, 32)
        self.num_classes = 20

    def prepare_data(self) -> None:
        self.train_dataset = VOCSegmentation(
            self.data_dir,
            year="2007",
            image_set="train",
            download=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.val_dataset = VOCSegmentation(
            self.data_dir,
            year="2007",
            image_set="val",
            download=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.test_dataset = VOCSegmentation(
            self.data_dir,
            year="2007",
            image_set="test",
            download=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def setup(self, stage: str) -> None:
        return super().setup(stage)

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
            # drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # drop_last=True,
        )
