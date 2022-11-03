import random
from tqdm import tqdm

import pandas as pd
import numpy as np

import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from albumentations.pytorch import ToTensorV2

def train_get_transforms(img_size):
    return A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    
                    
                    # A.ShiftScaleRotate(p=0.5),
                    # A.Rotate(limit=90, interpolation=1, border_mode=4, always_apply=False, p=0.5),
                    # A.Transpose(p=0.5),
                    # A.OneOf([
                    #         A.MotionBlur(blur_limit=5),
                    #         A.MedianBlur(blur_limit=5),
                    #         A.GaussianBlur(blur_limit=5),
                    #         A.GaussNoise(var_limit=(5.0, 30.0))], p=0.5),
                    
                    # A.OneOf([
                    #     A.GridDropout(),
                    #     A.Cutout(max_h_size=int(img_size * 0.5), max_w_size=int(img_size * 0.5), num_holes=1)], p=0.8),
                    # A.Cutout(max_h_size=int(img_size * 0.5), max_w_size=int(img_size * 0.5), num_holes=1, p=0.8),

                    ToTensorV2()
                    ])

def valid_get_transforms(img_size):
    return A.Compose([
                    A.Resize(img_size, img_size),


                    ToTensorV2()
                    ])


class BC_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.img_path = df['img_path']
        self.labels = df['N_category']
        self.transform = transform
        self.imgs = []

        for img_path in tqdm(self.img_path):
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

    def __len__(self): 
        return len(self.img_path)

    def __getitem__(self, idx):
        # img = cv2.imread(self.img_path[idx], cv2.COLOR_BGR2RGB)
        img = self.imgs[idx]
        label = self.labels[idx]

        if self.transform!=None:
            img = self.transform(image=img)['image']

        data = {
                'img' : torch.tensor(img, dtype=torch.float32),
                'label' : torch.tensor(label, dtype=torch.long),
                }

        return data