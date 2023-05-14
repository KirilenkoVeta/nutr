import torch
import wandb
from torch import nn
import pytorch_lightning as pl
from torchvision.utils import make_grid

from .encoder import Encoder


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

        # Log images with predicted and ground-truth nutrients
        if batch_idx == 0:
            # Get the first 8 images in the batch
            images = x[:8]
            # Get the predicted and ground-truth nutrients for the first 8 images
            pred_nutr_info = pred[:8]
            true_nutr_info = y[:8]
            # Create a grid of the images with their predicted and ground-truth nutrients
            grid = make_grid(images, nrow=4, normalize=True, scale_each=True)
            grid = wandb.Image(grid.permute(1, 2, 0).cpu().numpy())
            # Create a dictionary that maps nutrient names to their values
            nutrient_names = ['kcal_100', 'mass', 'prot_100', 'fat_100', 'carb_100']
            pred_nutr_dict = [{name: value.item() for name, value in zip(nutrient_names, sample)} 
                              for sample in pred_nutr_info]
            true_nutr_dict = [{name: value.item() for name, value in zip(nutrient_names, sample)} 
                              for sample in true_nutr_info]
            # Log the grid of images with their predicted and ground-truth nutrients
            wandb.log({'predicted_nutrients': wandb.Table(data=pred_nutr_dict),
                       'true_nutrients': wandb.Table(data=true_nutr_dict),
                       'images': grid})
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
