import argparse
import os
from pathlib import Path

import lightning as L
import torch
import torchvision
from lightning.pytorch.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from src.datasets.seg import SegmentationDataModule
from src.network.mae import MAEUNet
from src.network.seg import SegmentationModel


def get_args():
    """
    Parse arguments.
    """

    # fmt:off
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Set global seed")
    parser.add_argument("--name", type=str, default=None, help='Unique identifier for logs')

    parser.add_argument("-s", "--log_dir", type=str, default='/home/kkaai/scratch/saved', help='save path')
    parser.add_argument("--fast_dev_run", action='store_true', help="For debugging")
    parser.add_argument("--use_profiler", action='store_true', help="For debugging")

    # Datamodule
    parser.add_argument("--data_dir", default='/home/kkaai/scratch/data', type=str, help='root of training data.')
    parser.add_argument("--pretrain_path", default='', type=str, help='Path to pre-trained weights')
    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=224)
    parser.add_argument("--img_c", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64, help='Batch size to train with')
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers')

    # Training/Model
    parser.add_argument("--epochs",type=int, required=True, help='Number of pretraining epochs')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning rate used during training') 
    parser.add_argument("--weight_decay", type=float, default=0.05, help='weight decay of optimizer')  ## from centralai codebase

    args = parser.parse_args()
    # fmt:on

    #
    # Validate
    #
    args.data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    args.log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    assert os.path.isdir(args.data_dir), "Not found {}".format(args.data_dir)
    assert os.path.isdir(args.log_dir), "Not found {}".format(args.log_dir)

    return args


class GenerateCallback(Callback):
    def __init__(self, imgs, masks, every_n_epochs=5):
        super().__init__()
        self.imgs = imgs
        self.masks = masks
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, model):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            imgs = self.imgs.to(model.device)
            masks = self.masks.to(model.device)

            with torch.no_grad():
                model.eval()
                pred_masks = model(imgs)
                pred_masks = pred_masks.argmax(dim=1)
                model.train()
            # Plot and add to tensorboard
            masks = masks.unsqueeze(1).repeat(1,3,1,1)
            pred_masks = pred_masks.unsqueeze(1).repeat(1,3,1,1)
            imgs = torch.stack([imgs, masks, pred_masks], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3, normalize=True)
            trainer.logger.experiment.add_image(
                "Segmentations", grid, global_step=trainer.global_step
            )


if __name__ == "__main__":
    args = get_args()
    print("Command line:", vars(args))
    L.seed_everything(args.seed)

    #
    # Get data module
    #
    data_module = SegmentationDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        img_c=args.img_c,
    )
    data_module.setup()
    print("Created data_module")

    #
    # Get pre-training model
    #
    model = SegmentationModel(
        backbone="resnet18",
        num_classes=data_module.train_dataset.num_classes,
        img_c=args.img_c,
    )

    #
    # Get pre-trained weights
    #
    if args.pretrain_path:
        pretrained_model = torch.load(args.pretrain_path)
        print("Pretrained_path:", args.pretrain_path, type(pretrained_model))
        assert isinstance(pretrained_model, MAEUNet)
        model.model.encoder = pretrained_model.encoder

    #
    # Example images for callback
    #
    num = 10
    example_imgs = [data_module.train_dataset[i] for i in range(num)]
    imgs = torch.stack([example_imgs[i][0] for i in range(num)], dim=0)
    masks = torch.stack([example_imgs[i][1] for i in range(num)], dim=0)

    #
    # Setup callbacks
    #
    model.name = f"seg_{Path(args.pretrain_path).stem}"
    lr_callback = LearningRateMonitor("epoch")
    custom_callback = GenerateCallback(imgs=imgs, masks=masks)

    # setup tensorboard logging
    logger_dir = os.path.join(args.log_dir, "syde_logs", model.name)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, log_graph=True)

    weights_dir = os.path.join(args.log_dir, "syde_weights", model.name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_dir,
        save_top_k=1,
        filename="{step}-{loss:.2f}",
        monitor="seg/val/iou",
        mode="max",
    )

    #
    # setup pytorch lightning trainer
    #
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        logger=tb_logger,
        profiler=SimpleProfiler() if args.use_profiler else None,
        callbacks=[custom_callback, checkpoint_callback, lr_callback],
        fast_dev_run=args.fast_dev_run,
    )
    print("Setup trainer")

    trainer.fit(model, data_module)
    trainer.validate(model, data_module, ckpt_path="best")
