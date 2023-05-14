import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FoodDataset(Dataset):

    def __init__(self, 
                path_to_csv: str = './dataset/train/train.csv', 
                path_to_imgs: str = './dataset/train', 
                is_read_all: bool = True, 
                scale: int = 128):
        self.nutr_info = pd.read_csv(path_to_csv)
        self.path_to_imgs = path_to_imgs
        self.is_read_all = is_read_all
        self.scale = scale
        if self.is_read_all:
            self.images = [self.transform(Image.open(f'{self.path_to_imgs}/{i}.png')) 
                           for i in self.nutr_info.img]

    def __len__(self):
        return len(self.nutr_info)
    
    def transform(self, img):
        size = min(img.size)
        return transforms.Compose([transforms.CenterCrop(size), 
                                   transforms.Resize(self.scale), 
                                   transforms.ToTensor()])(img)
    
    def __getitem__(self, idx):
        sample = dict(self.nutr_info[['kcal_100', 'mass', 'prot_100', 
                                      'fat_100', 'carb_100']].iloc[idx])
        if self.is_read_all:
            image = self.images[idx]
        else:
            image = self.transform(Image.open(
                f'{self.path_to_imgs}/{self.nutr_info.img.iloc[idx]}.png'))
        sample['image'] = image
        sample['nutr_info'] = torch.tensor(list(sample.values()))
        return sample
