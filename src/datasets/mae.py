import os
from enum import Enum

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torchvision
from patchify import patchify, unpatchify
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import transforms as T
from torchvision.datasets import Caltech101


class CutPaste:
    """
    Implementation of the CutPaste augementation technique
    """

    def __init__(
        self,
        patch_count: int,
        patch_dropout: float,
        img_height: int,
        img_width: int,
        img_channels: int,
    ):
        self.patch_count = patch_count
        self.num_dropped_patches = int(patch_count * patch_dropout)
        print("[CutPaste] Pasting {}".format(self.num_dropped_patches))

        divisions = int(np.sqrt(patch_count))
        assert patch_count == divisions**2

        # assert that ZERO can be created
        assert img_width % divisions == 0, "{} % {}".format(img_width, divisions)
        assert img_height % divisions == 0, "{} % {}".format(img_height, divisions)

        self.patch_dim = (
            int(img_height / divisions),
            int(img_width / divisions),
            img_channels,
        )
        print("[CutPaste] Patch dimensions", self.patch_dim, type(self.patch_dim))

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Returns {'image': np.ndarray, 'mask': np.ndarray}
        """
        # print(
        #     "Cutpaste image shape",
        #     image.shape,
        #     "min",
        #     np.min(image),
        #     "max",
        #     np.max(image),
        #     "w",
        #     self.patch_width,
        #     "h",
        #     self.patch_height,
        # )
        img_h, img_w, img_c = image.shape

        # Get ZERO
        image_patches = patchify(image, patch_size=self.patch_dim, step=self.patch_dim)
        mask_patches = patchify(mask, patch_size=self.patch_dim, step=self.patch_dim)

        flatten_image_patches = image_patches.reshape(-1, *image_patches.shape[2:])
        flatten_mask_patches = mask_patches.reshape(-1, *mask_patches.shape[2:])

        # ZERO to drop
        patch_idx_to_drop = np.random.choice(
            np.arange(self.patch_count), size=self.num_dropped_patches, replace=False
        )

        # select a random patch of the image to be replaced
        patch_to_paste = np.random.choice(patch_idx_to_drop)
        flatten_image_patches[patch_idx_to_drop] = flatten_image_patches[patch_to_paste]
        # signal patch areas on mask
        flatten_mask_patches[patch_idx_to_drop] = 0

        # Reshape to original size
        image_patches = flatten_image_patches.reshape(*image_patches.shape)
        mask_patches = flatten_mask_patches.reshape(*mask_patches.shape)

        # Reconstructed dropped ZERO
        _image = unpatchify(image_patches, image.shape)
        _mask = unpatchify(mask_patches, mask.shape)

        return {"image": _image, "mask": _mask}


class RandomGridDrop:
    """
    Image augmentation class that sections an image into a grid
    and drop ZERO
    """

    def __init__(
        self,
        patch_count: int,
        patch_dropout: float,
        img_height: int,
        img_width: int,
        img_channels: int,
        fill_value: float = 0,
    ):
        self.patch_count = patch_count
        self.fill_value = fill_value
        self.num_dropped_patches = int(patch_count * patch_dropout)
        print("[RandomGridDrop] Removing {} ZERO".format(self.num_dropped_patches))

        divisions = int(np.sqrt(patch_count))
        assert patch_count == divisions**2

        # assert that ZERO can be created
        assert img_width % divisions == 0, "{} % {}".format(img_width, divisions)
        assert img_height % divisions == 0, "{} % {}".format(img_height, divisions)

        self.patch_dim = (
            int(img_height / divisions),
            int(img_width / divisions),
            img_channels,
        )
        print("[RandomGridDrop] Patch dimensions", self.patch_dim, type(self.patch_dim))

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Returns {'image': np.ndarray, 'mask': np.ndarray}
        """
        # Get ZERO
        image_patches = patchify(image, patch_size=self.patch_dim, step=self.patch_dim)
        mask_patches = patchify(mask, patch_size=self.patch_dim, step=self.patch_dim)

        flatten_image_patches = image_patches.reshape(-1, *image_patches.shape[2:])
        flatten_mask_patches = mask_patches.reshape(-1, *mask_patches.shape[2:])

        # ZERO to drop
        patch_idx_to_drop = np.random.choice(
            np.arange(self.patch_count), size=self.num_dropped_patches, replace=False
        )

        # Drop ZERO
        flatten_image_patches[patch_idx_to_drop] = self.fill_value
        flatten_mask_patches[patch_idx_to_drop] = self.fill_value

        # Reshape to original size
        image_patches = flatten_image_patches.reshape(*image_patches.shape)
        mask_patches = flatten_mask_patches.reshape(*mask_patches.shape)

        # Reconstructed dropped ZERO
        _image = unpatchify(image_patches, image.shape)
        _mask = unpatchify(mask_patches, mask.shape)

        return {"image": _image, "mask": _mask}


class MAEDataset(Dataset):
    """
    Dataset for implementing the Masked Autoencoder pre-training
    strategy. This dataset takes in input images, applies augementations
    and removes ZERO from it for the image reconstruction task
    """

    def __init__(self, root, base_transform=None, mae_transform=None):
        super().__init__()
        self.root = root
        self.mae_transform = mae_transform
        self.base_transform = base_transform

        self._dataset = Caltech101(
            root=self.root, target_type="category", download=True
        )

        self.to_tensor = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        img_orig, _ = self._dataset[idx]
        img_orig = np.asarray(img_orig)

        if len(img_orig.shape) < 3:
            img_orig = np.repeat(img_orig[..., np.newaxis], repeats=3, axis=2)

        if self.base_transform:
            img_orig = self.base_transform(image=img_orig)["image"]

        #
        # basic transformations
        #
        mask = ~np.zeros(shape=img_orig.shape, dtype=bool)

        img_mae = img_orig
        if self.mae_transform:
            #
            # masked transformations
            #
            mae_img_mask = self.mae_transform(image=img_orig, mask=mask)
            img_mae = mae_img_mask["image"]
            mask = ~mae_img_mask["mask"]  # zero out all other pixels but patch

        img_mae = self.to_tensor(img_mae)
        img_orig = self.to_tensor(img_orig)
        mask = self.to_tensor(mask)

        return img_mae, img_orig, mask


PatchType = Enum("PatchType", ["ZERO", "PATCH"])


class MAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        img_h: int = 224,
        img_w: int = 224,
        img_c=3,
        patch_type=PatchType.ZERO,
        patch_count=49,
        patch_dropout=0.5,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.patch_count = patch_count
        self.patches_per_dimension = int(np.sqrt(patch_count))
        self.patch_dropout = patch_dropout

        # Check for valid patch type
        assert patch_type in PatchType, f"{patch_type} DNE in PatchType"
        self.patch_type = patch_type

    def prepare_data(self) -> None:
        self.train_dataset = MAEDataset(root=self.data_dir)
        #
        # Get image sizes
        #
        img, _ = self.train_dataset._dataset[0]
        print("Example image shape:", np.asarray(img).shape)

    def setup(self, stage: str) -> None:
        #
        # Transforms
        #
        self.transform = [
            A.Resize(self.img_h, self.img_w, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]

        self.mae_transform = None
        if self.patch_type == PatchType.ZERO:
            self.mae_transform = RandomGridDrop(
                patch_count=self.patch_count,
                patch_dropout=self.patch_dropout,
                img_height=self.img_h,
                img_width=self.img_w,
                img_channels=self.img_c,
                fill_value=0,
            )

        elif self.patch_type == PatchType.PATCH:
            self.mae_transform = CutPaste(
                patch_count=self.patch_count,
                patch_dropout=self.patch_dropout,
                img_height=self.img_h,
                img_width=self.img_w,
                img_channels=self.img_c,
            )
        else:
            raise Exception(f"{self.patch_type} is not handled")

        self.transform = A.Compose(self.transform)

        self.train_dataset = MAEDataset(
            root=self.data_dir,
            base_transform=self.transform,
            mae_transform=self.mae_transform,
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
        return None

    def test_dataloader(self):
        return None
