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
    transforms.Grayscale(num_output_channels=1),
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
            nn.Linear(64*20*20, 128),
            nn.ReLU(True),

            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        print(x.shape, ' input size')
        x = self.encoder(x)
        print(x.shape, ' conv layer')

        x = x.view(x.size(0), -1)
        print(x.shape, ' size change for linear layer')
        x = self.encoder_linear(x)
        print(x.shape, ' linear layer')
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(8),

            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(True),

            nn.Linear(128, 64*20*20),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        print('hey')
        print(x.shape, 'decoder input')
        x = self.decoder_linear(x)
        print(x.shape, 'decoder linear output')
        x = x.view(x.size(0),self.decoder[0].in_channels,20,20
                   )
        print(x.shape, 'decoder shape change output')

        x = self.decoder(x)
        return x

# Creating Model objects & parameters
encoder = ConvEncoder()
decoder = ConvDecoder()
parameters = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

# Hyper parameters
optimizer = torch.optim.Adam(parameters, lr=0.05, weight_decay=1e-15)
crit = torch.nn.MSELoss()
epchs = 5

def train_epoch(encoder, decoder, crit, optimizer):
    
    correct = []
    total = []
    for epoch,(image,label) in enumerate(train_loader):
        
        encoded = encoder(image)
        decoded = decoder(encoded)

        print(image.shape, ' image img')
        print(decoded.shape, ' decoded img')

        loss = crit(decoded, image)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

train_epoch(encoder, decoder, crit, optimizer)

#testing
# Testing phase
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

img = Image.open('/content/smple.png')
img = transform(img)
img = torch.unsqueeze(img, 0)  # add a new dimension at the beginning

embedd = encoder(img)
print(embedd.shape)
generated = decoder(embedd)
print(generated.shape)
