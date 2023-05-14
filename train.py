import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import argparse

from food_dataset import FoodDataset
from nutr_pred import NutrPred, NutrientsLogger


def main(args):
    train_dataset = FoodDataset(
        is_read_all=True,
        scale=args.scale
    )
    val_dataset = FoodDataset(
        path_to_csv='./dataset/val/val.csv',
        path_to_imgs='./dataset/val',
        is_read_all=True,
        scale=args.scale
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    samples = next(iter(val_loader))

    wandb_logger = WandbLogger(
        project='nutr_pred',
        name=args.name
    )
    wandb_logger.log_hyperparams(args)

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1,
        max_epochs=args.epochs,
        callbacks=[NutrientsLogger(samples)]
    )
    model = NutrPred(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        downsample_steps_before=args.downsample_steps_before,
        conv_steps=args.conv_steps,
        downsample_steps_after=args.downsample_steps_after,
        dropout=args.dropout,
        mlp_hidden=args.mlp_hidden,
        lr=args.lr
    )
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--downsample_steps_before', type=int, default=4)
    parser.add_argument('--conv_steps', type=int, default=1)
    parser.add_argument('--downsample_steps_after', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--mlp_hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--scale', type=int, default=128)
    parser.add_argument('--name', type=str, default='default')
    args = parser.parse_args()
    main(args)
