import torch
import wandb
from torch import nn
import pytorch_lightning as pl
from torchvision.utils import make_grid

from encoder import Encoder


class NutrPred(pl.LightningModule):
    def __init__(self, 
                in_channels=3, 
                hidden_channels=64, 
                downsample_steps_before=4, 
                conv_steps=1, 
                downsample_steps_after=2,
                dropout=0.20,
                mlp_hidden=128,
                lr=0.0003):
        super().__init__()
        self.save_hyperparameters()

        # module that encodes an image into a featuremap
        self.encoder = Encoder(
            in_channels, 
            hidden_channels, 
            downsample_steps_before, 
            conv_steps, 
            downsample_steps_after,
            dropout
        )

        # module that flattens a featuremap into vector and predicts nutrients
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 2 * hidden_channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 5)
        )
        self.lr = lr
        self.loss_function = nn.L1Loss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['nutr_info']
        pred = self(x)
        loss = self.loss_function(pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['nutr_info']
        pred = self(x)
        loss = self.loss_function(pred, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['nutr_info']
        pred = self(x)
        loss = self.loss_function(pred, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class NutrientsLogger(pl.Callback):
    def __init__(self, val_batch, num_samples=30) -> None:
        super().__init__()
        x, y = val_batch['image'], val_batch['nutr_info']
        self.x = x[:num_samples]
        self.y = y[:num_samples]


    def on_validation_batch_end(self, trainer, pl_module):
        images = self.x
        pred_nutr_info = pl_module(images)
        true_nutr_info = self.y

        columns = ['Input image', 'Prediction', 'Expected']
        data = [
            [
                wandb.Image(images[idx].permute(1, 2, 0).cpu().numpy()), 
                pred_nutr_info[idx].tolist(),
                true_nutr_info[idx].tolist()
            ] 
            for idx in range(8)
        ]

        # Log the grid of images with their predicted and ground-truth nutrients
        table = wandb.Table(columns=columns, data=data)
        trainer.logger.experiment.log({'nutrients': table})
