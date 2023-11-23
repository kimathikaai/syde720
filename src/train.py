import lightning as L
import argparse
import datetime
import os

def get_args():
    """
    Parse arguments.
    """
    # fmt:off
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Set global seed")
    parser.add_argument("--name", type=str, default=None, help='Unique identifier for logs')

    parser.add_argument("-r", "--data_dir", type=str, help='root of training data.')
    parser.add_argument("-s", "--log_dir", type=str, help='save path')
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers')
    parser.add_argument("--fast_dev_run", action='store_true', help="For debugging")
    parser.add_argument("--use_profiler", action='store_true', help="For debugging")

    parser.add_argument("--epochs",type=int, help='Number of pretraining epochs')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size to train with')
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

    
