import argparse
import os

import lightning as L
from numpy import who
import torch
import torchvision
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from src.datasets.mae import MAEDataModule, PatchType
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
    parser.add_argument("--pretrain-ckpt", type=str, help='Path to pre-trained weights')
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

    # convert into enum
    args.patch_type = PatchType[args.patch_type]

    return args


class GenerateCallback(Callback):
    """
    During training, we want to keep track of the learning progress by seeing
    reconstructions made by our model. For this we implement a callback object
    which will add reconstructions every N epochs to our tensorboard
    """

    def __init__(self, img_orig, img_mae, mask, every_n_epochs=1):
        super().__init__()
        self.img_mae = img_mae
        self.img_orig = img_orig
        self.mask = mask
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            img_orig = self.img_orig.to(pl_module.device)
            img_mae = self.img_mae.to(pl_module.device)
            mask = self.mask.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(img_mae)
                pl_module.train()

            # Plot and add to tensorboard
            imgs = torch.stack([reconst_imgs, img_mae, img_orig, mask], dim=1).flatten(
                0, 1
            )
            grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
            trainer.logger.experiment.add_image(
                "Reconstructions", grid, global_step=trainer.global_step
            )


if __name__ == "__main__":
    args = get_args()
    L.seed_everything(args.seed)

    #
    # Get data module
    #
    data_module = MAEDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        img_c=args.img_c,
        patch_type=args.patch_type,
        patch_count=args.patch_count,
        patch_dropout=args.patch_dropout,
    )
    data_module.prepare_data()
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    print("Created data_module")

    #
    # Get pre-training model
    #
    model = SegmentationModel(
        backbone="resnet18",
        num_classes=len(data_module.train_dataset._dataset.categories),
        img_h=args.img_h,
        img_w=args.img_w,
        img_c=args.img_c,
    )

    #
    # Example images for callback
    #
    num = 3
    example_imgs = [data_module.train_dataset[i] for i in range(num)]
    img_mae = torch.stack([example_imgs[i][0] for i in range(num)], dim=0)
    img_orig = torch.stack([example_imgs[i][1] for i in range(num)], dim=0)
    mask = torch.stack([example_imgs[i][2] for i in range(num)], dim=0)

    #
    # Setup callbacks
    #
    model.name = (
        f"mae_{args.patch_type}_{args.patch_count}_{args.patch_dropout}_{args.epochs}"
    )
    generate_callback = GenerateCallback(
        img_orig=img_orig,
        mask=mask,
        img_mae=img_mae,
    )
    lr_callback = LearningRateMonitor("epoch")

    # setup tensorboard logging
    logger_dir = os.path.join(args.log_dir, "syde_logs", model.name)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, log_graph=True)

    #
    # setup pytorch lightning trainer
    #
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        logger=tb_logger,
        profiler=SimpleProfiler() if args.use_profiler else None,
        callbacks=[generate_callback, lr_callback],
        fast_dev_run=args.fast_dev_run,
        val_check_interval=0,
    )
    print("Setup trainer")

    #
    # train
    #
    trainer.fit(model, train_dataloaders=train_dataloader)

    #
    # Save pre-trained model
    #
    torch.save(
        model, os.path.join(args.log_dir, "syde_logs", "{}.ckpt".format(model.name))
    )
