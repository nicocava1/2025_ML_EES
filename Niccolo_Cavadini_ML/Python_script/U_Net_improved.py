import os
import rasterio
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
# ================================
# MODEL
# ================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3,64)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64,128)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128,256)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256,512)

        self.u1 = nn.ConvTranspose2d(512,256,2,2)
        self.c1 = DoubleConv(512,256)

        self.u2 = nn.ConvTranspose2d(256,128,2,2)
        self.c2 = DoubleConv(256,128)

        self.u3 = nn.ConvTranspose2d(128,64,2,2)
        self.c3 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3)

        x = self.u1(d4); x = self.c1(torch.cat([x,d3],1))
        x = self.u2(x);  x = self.c2(torch.cat([x,d2],1))
        x = self.u3(x);  x = self.c3(torch.cat([x,d1],1))

        return torch.sigmoid(self.out(x))
