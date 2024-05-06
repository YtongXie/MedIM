import datetime
import os
from argparse import ArgumentParser

import segmentation_models_pytorch as smp
import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from medim.datasets.data_module import DataModule
from medim.datasets.segmentation_dataset import SIIMImageDataset
from medim.models.backbones.transformer_seg import SETRModel
from medim.models.medim.medim_train import MedIM
from medim.models.ssl_segmenter import SSLSegmenter

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser(
        "Finetuning of semantic segmentation task for MedIM")
    parser.add_argument("--base_model", type=str,
                        default="vit", help="resnet50 or vit")
    parser.add_argument("--ckpt_path", type=str,
                        default="None")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_pct", type=float, default=1)
    parser.add_argument("--outpath", type=str, default="tmp")
    parser.add_argument("--outdir", type=str, default="tmp")

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.max_epochs = 50

    seed_everything(args.seed)

    datamodule = DataModule(SIIMImageDataset, None,
                            None, args.data_pct,
                            args.batch_size, args.num_workers)

    if args.ckpt_path != 'None':
        medim = MedIM.load_from_checkpoint(args.ckpt_path)
    else:
        medim = MedIM(**args.__dict__)
    encoder = medim.img_encoder_q.model
    decoder = medim.img_decoder_q.model

    if args.base_model == "vit":
        args.seg_model = SETRModel(
            patch_size=(16, 16),
            in_channels=3,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64]
        )
        args.seg_model.encoder_2d.bert_model = encoder
        args.seg_model.decoder_2d.decoder_1 = decoder.decoder_1
        args.seg_model.decoder_2d.decoder_2 = decoder.decoder_2
        args.seg_model.decoder_2d.decoder_3 = decoder.decoder_3
        args.seg_model.decoder_2d.decoder_4 = decoder.decoder_4

        # for param in args.seg_model.encoder_2d.bert_model.parameters():
        #     param.requires_grad = False

    elif args.base_model == "resnet50":
        # FIXME: fix this later
        args.seg_model = smp.Unet(
            args.base_model, encoder_weights=None, activation=None)

        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path)
            ckpt_dict = dict()
            for k, v in ckpt["state_dict"].items():
                if k.startswith("img_encoder_q.model"):
                    new_k = ".".join(k.split(".")[2:])
                    new_k = new_k.replace("blocks", "layer")
                    ckpt_dict[new_k] = v

            ckpt_dict["fc.bias"] = None
            ckpt_dict["fc.weight"] = None

            args.seg_model.encoder.load_state_dict(ckpt_dict)
            # Freeze encoder
            for param in args.seg_model.encoder.parameters():
                param.requires_grad = False

    model = SSLSegmenter(**args.__dict__)

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
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project=args.outdir, save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')

if __name__ == "__main__":
    cli_main()