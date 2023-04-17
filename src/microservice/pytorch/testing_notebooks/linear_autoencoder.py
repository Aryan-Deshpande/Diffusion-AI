# loss in the neural network architecture is not good, does not reconstruct as well ( its just noisy images )

#torch architecture imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as t
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
trans = transforms.Compose([
    transforms.ToTensor(),
])

image = Image.open('./test.png').convert('L')
image = trans(image)
print(image.shape, " aft trans")
# open image dataset
##load = torchvision.datasets.ImageFolder(root='C://Users//deshp//Desktop//Dp//src//pytorch//test1//test1')
##testing = DataLoader(batch_size=64, shuffle=True, dataset=load)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # reduce input image size
            nn.Linear(100*96, 3200),
            nn.ReLU(),

            nn.Linear(3200, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 12),
            nn.ReLU(),

            nn.Linear(12, 3),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),

            nn.Linear(12, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.ReLU(),      

            nn.Linear(512, 3200),
            nn.ReLU(), 

            nn.Linear(3200, 100*96),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        #print(decoded)

        return decoded

model = Autoencoder()
print(model.encoder[0].weight.shape)

# training
# hyperparameters

optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-3)
crit = nn.MSELoss()
epochs = 100

# viewing architecture
from torch.utils.tensorboard import SummaryWriter

logs = 'logs/'
writer = SummaryWriter(logs)


output = []
for epoch in range(100):
    x = image.view(image.size(0), -1)
    recons = model(x)

    #print(recons)

    if epoch % 10 == 0:    
        output.append(recons)

    loss = crit(recons, x)

    loss.backward()
    writer.add_scalar('training_loss', loss.item(), epoch)
    print(f"{loss.item()}")

    optimizer.step()
    optimizer.zero_grad()

writer.add_graph(model, input_to_model=image.view( image.size(0),-1) )

print(output)

for epoch in range(10):
    plt.figure(figsize=(9,2))
    plt.gray()

    recon = output[epoch].detach().numpy()

    for i,item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2,9,9+i+1)
        item = item.reshape(-1,100,96)

        plt.imshow(item[0])



