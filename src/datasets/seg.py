import os
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Food103Dataset(Dataset):
    def __init__(self, root, stage, transform=None, binary_class=True) -> None:
        super().__init__()

        assert stage in ["train", "test"]

        self.root = root
        self.transform = transform
        self.to_tensor = T.Compose([T.ToTensor()])

        self.imgs = sorted(
            glob(os.path.join(self.root, "img_dir", stage, "*")),
            key=lambda x: Path(x).stem,
        )
        self.masks = sorted(
            glob(os.path.join(self.root, "ann_dir", stage, "*")),
            key=lambda x: Path(x).stem,
        )
        assert len(self.imgs) > 0
        assert len(self.masks) > 0
        assert len(self.imgs) == len(self.masks)
        print("Images:", len(self.imgs))

        self.binary_task = binary_class
        if self.binary_task:
            self.num_classes = 103
        else:
            self.num_classes = 2

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        #
        # opencv reads images in BGR format by default. Need to switch for transforms
        #
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx])

        # do augmentation here
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        if self.binary_task:
            mask[mask > 0] = 1

        #
        # T.ToTensor normalizes the image.
        #
        img = self.to_tensor(img)
        mask = torch.from_numpy(mask).long()

        return img, mask


class SegmentationDataModule(L.LightningDataModule):
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

        self.num_classes = None

    def setup(self, stage=None) -> None:
        #
        # Transforms
        #
        self.transform_train = A.Compose(
            [
                A.Resize(self.img_h, self.img_w, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.GridDistortion(p=0.2),
                A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                A.GaussNoise(),
            ]
        )
        self.transform_val = A.Compose(
            [
                A.Resize(self.img_h, self.img_w, interpolation=cv2.INTER_NEAREST),
            ]
        )

        self.train_dataset = Food103Dataset(
            root=self.data_dir, stage="train", transform=self.transform_train
        )
        self.val_dataset = Food103Dataset(
            root=self.data_dir, stage="test", transform=self.transform_val
        )

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
