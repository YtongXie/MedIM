import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from medim.datasets.classification_dataset import (CheXpert5ImageDataset,
                                                  CheXpert14ImageDataset,
                                                  COVIDXImageDataset,
                                                  VinDrImageDataset)
from medim.datasets.data_module import DataModule
from medim.datasets.transforms import DataTransforms, Moco2Transform
from medim.models.medim.medim_train import MedIM
from medim.models.ssl_finetuner import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert5")
    parser.add_argument("--path", type=str, default="None")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--outpath", type=str, default="tmp")
    parser.add_argument("--outdir", type=str, default="tmp")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--accumulate", type=int, default=1, help="number of accumulate_grad_batches")

    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_epochs = args.epoch
    args.accumulate_grad_batches = args.accumulate

    seed_everything(args.seed)

    if args.dataset == "chexpert5":
        # define datamodule
        # check transform here
        datamodule = DataModule(CheXpert5ImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 5
        multilabel = True
    elif args.dataset == "chexpert14":
        datamodule = DataModule(CheXpert14ImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 14
        multilabel = True          
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    elif args.dataset == "vindr":
        datamodule = DataModule(VinDrImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 28
        multilabel = True
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    if args.path != "None":
        model = MedIM.load_from_checkpoint(args.path, strict=False)
    else:
        model = MedIM(**args.__dict__)

    args.model_name = model.hparams.img_encoder
    args.backbone = model.img_encoder_q
    args.in_features = args.backbone.feature_dim
    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    # extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = args.outpath
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/{args.outdir}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=50, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    # extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data/wandb")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project=args.outdir, save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
