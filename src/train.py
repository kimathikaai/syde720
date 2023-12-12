import argparse
import os

import lightning as L

from src.datasets.mae import MAEDataModule, PatchType


def get_args():
    """
    Parse arguments.
    """
    # fmt:off
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Set global seed")
    parser.add_argument("--name", type=str, default=None, help='Unique identifier for logs')

    parser.add_argument("-s", "--log_dir", type=str, help='save path')
    parser.add_argument("--fast_dev_run", action='store_true', help="For debugging")
    parser.add_argument("--use_profiler", action='store_true', help="For debugging")

    # Datamodule
    parser.add_argument("-r", "--data_dir", type=str, help='root of training data.')
    parser.add_argument("--patch_type", type=str, default=PatchType.ZERO.name, choices=[p.name for p in PatchType])
    parser.add_argument("--patch_count", type=int, default=49)
    parser.add_argument("--patch_dropout", type=float, default=0.5)
    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=224)
    parser.add_argument("--img_c", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64, help='Batch size to train with')
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers')

    # Training/Model
    parser.add_argument("--epochs",type=int, help='Number of pretraining epochs')
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


def pretraining(model: L.LightningModule, args) -> L.LightningModule:
    """
    Perfom the masked autoencoding pre-training
    """

    return model


def downstream(model: L.LightningModule, args) -> None:
    pass


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
