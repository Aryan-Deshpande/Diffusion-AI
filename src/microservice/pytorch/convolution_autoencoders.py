#torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv
import torchvision.transforms as t
from torchvision import transforms
from  torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Data & Preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

train_loader = DataLoader(dataset=MNIST(root='../trainmnist',train=True, download=True, transform=transform), shuffle=True, batch_size=15)

# Deep learning architecture
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

        ) # flatten -> 

        self.encoder_linear = nn.Sequential(
            
        )
    
    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.decoder = nn.Sequential(

        )

        self.decoder_linear = nn.Sequential(
            
        )
    
    def forward(self, x):
        pass


encoder = ConvEncoder()
decoder = ConvDecoder()
parameters = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

# Hyper parameters
optimizer = torch.optim.Adam(parameters, lr=0.05, weight_decay=1e-15)
crit = torch.nn.MSELoss()
epchs = 100


def train_epoch(encoder, decoder, crit, optimizer):
    
    correct = []
    total = []
    for epoch,(image,label) in enumerate(train_loader):
        pass
        
        encoded = encoder(image)
        decoded = decoder(encoded)

        loss = crit(decoded, image)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # calculates the accuracy
        total += image
        correct+= (decoded == image).sum.item()

        with torch.no_grad():
            if epoch % 10 == 0:
                print(f'Accuracy: {100*(correct/total)}%, Loss: {loss.item()}')

# Testing phase
img = Image.open('./aryan8.png')
img = transform(img)

def test():
    model = ConvEncoder()
    predicted = model(img)

test()