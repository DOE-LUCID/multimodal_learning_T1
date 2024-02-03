import os

import numpy as np
import pandas as pd
from pdb import set_trace
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms


class custom_dataloader(Dataset):
    def __init__(self, opt, data, split=None, mode='wsi'):
        print("------------------- data[split].keys()----------------", data[split].keys())
        self.X_wsi = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        # self.X_omic = self.X_omic
        self.grade = data[split]['g'] # grade

        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_wsi),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __getitem__(self, index):
        grade = torch.tensor(self.grade[index]).type(torch.LongTensor)

        X_wsi = Image.open(self.X_wsi[index]).convert('RGB')
        X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
        # set_trace()
        return (self.transforms(X_wsi), X_omic, grade)

    def __len__(self):
        return len(self.X_wsi)



