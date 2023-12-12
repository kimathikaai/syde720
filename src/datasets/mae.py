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

    def __init__(self, patch_width, patch_height):
        self.patch_width = patch_width
        self.patch_height = patch_height

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

        # crop = T.RandomCrop((self.patch_width, self.patch_height))
        # Get random patch
        x_pos_patch = np.random.randint(0, img_w - self.patch_width)
        y_pos_patch = np.random.randint(0, img_h - self.patch_height)
        # print("x_pos_patch", x_pos_patch, "y_pos_patch", y_pos_patch)
        image_patch = image[
            y_pos_patch : y_pos_patch + self.patch_height,
            x_pos_patch : x_pos_patch + self.patch_width,
            :,
        ]
        # image_patch = self.crop(image)

        # Get pasting position
        x_pos = np.random.randint(0, img_w - self.patch_width)
        y_pos = np.random.randint(0, img_h - self.patch_height)
        # print("x_pos", x_pos, "y_pos", y_pos)

        # Apply crop to original image
        patch_image = image.copy()
        patch_image[
            y_pos : y_pos + self.patch_height, x_pos : x_pos + self.patch_width, :
        ] = image_patch

        # 1 = location where the patch was pasted
        # 0.5 = location where the patch was sourced
        mask[y_pos : y_pos + self.patch_height, x_pos : x_pos + self.patch_width, :] = 0
        # mask[y_pos_patch : y_pos_patch + self.patch_height, x_pos_patch : x_pos_patch + self.patch_width, :] = 0
        # mask[:, x_pos_patch:x_pos_patch+self.patch_width, y_pos_patch:y_pos_patch+self.patch_height] = 0.5

        return {"image": patch_image, "mask": mask}


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

    def __init__(self, data_dir, images_list, base_transform, mae_transform):
        super().__init__()
        self.data_dir = data_dir
        self.images_list = images_list
        self.mae_transform = mae_transform

        self._dataset = Caltech101(
            root=self.data_dir,
            target_type="category",
            download=True,
            transform=base_transform,
        )

        self.to_tensor = T.Compose([T.ToTensor()])  # , T.Normalize(self.mean, self.std)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        img_orig, _ = self._dataset[idx]

        #
        # basic transformations
        #
        mask = ~np.zeros(shape=img_orig.shape, dtype=bool)

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


PatchType = Enum("PatchType", ["ZERO", "IMAGE"])


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

        self.num_classes = 20
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
        self.train_dataset = Caltech101(
            root=self.data_dir, target_type="category", download=True
        )
        #
        # Get image sizes
        #
        img, _ = self.train_dataset[0]
        import pdb

        pdb.set_trace()
        self.img_h = img.size[0]
        self.img_w = img.size[1]
        self.img_c = img.size[0]

    def setup(self, stage: str) -> None:
        #
        # Transforms
        #
        self.transform = [
            A.Resize(self.img_h, self.img_w, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]

        if self.patch_type == PatchType.ZERO:
            self.transform.append(
                RandomGridDrop(
                    patch_count=self.patch_count,
                    patch_dropout=self.patch_dropout,
                    img_height=self.img_h,
                    img_width=self.img_w,
                    img_channels=self.img_c,
                    fill_value=0,
                ),
            )
        elif self.patch_type == PatchType.IMAGE:
            self.transform.append(
                CutPaste(
                    patch_width=self.img_w // self.patches_per_dimension,
                    patch_height=self.img_h // self.patches_per_dimension,
                )
            )
        else:
            raise Exception(f"{self.patch_type} is not handled")

        self.transform = A.Compose(self.transform)

        self.train_dataset = Caltech101(
            root=self.data_dir,
            target_type="category",
            download=True,
            transform=self.transform,
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
