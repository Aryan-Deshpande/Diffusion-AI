import torch 
import torch.nn as nn

from torchvision import transforms

# backward
import torch.nn as nn

class SPE(nn.Module):
  def __init__(self):
    super(SPE, self).__init__()

  def forward(self):
    pass


class Block(nn.Module):
  def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super(Block, self).__init__()
    
    self.time_mlp = nn.Linear(time_emb_dim, out_ch) # map the time embedding 
                                                    # to the out embed

    if up == True:
      self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
      self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
    else:
      self.conv1 = nn.Conv2d(2*out_ch, in_ch, 3, padding=1)
      self.transform = nn.ConvTranspose(out_ch, out_ch, 4, 2, 1)

    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.pool = nn.MaxPool2d(3, stride=2)
    self.bnorm = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU()
  
  def forward(self, x, t, ):
    # first conv layer
    h = self.bnorm(self.relu(self.conv1(x) )) 

    # time embedding
    time_emb = self.relu(self.time_mlp(t))

    # extend last 2 dim ?
    # here what happens is [(..., )] specifies to retain the existing dimensions
    # + (None, ) * 2 specifies to ad 2 dimensions
    time_emb = time_emb[(..., ) + (None, ) * 2]

    # add the time channel to the input
    h = h + time_emb

    # second conv layer
    h = self.bnorm(self.relu(self.conv2(h)))

    # down or upsample based on the arguments in the constructor of the class
    return self.transform(h)

class unet(nn.Module):
  def __init__(self):
    super(unet,self).__init__()
    image_channels = 3
    down_channels = (64,128,256,512,1024)
    up_channels = (1024, 512, 256, 128, 64)

    out_dim = 1
    time_emb_dim = 32

    # Time embedding
    self.time_mlp = nn.Sequential(
        SinusodialPositionEmbeddings(time_emb_dim),
        nn.Linear(time_emb_dim, time_emb_dim),
        nn.ReLU()
    )

    self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

    self.downs = nn.ModuleList([Block( down_channels[i], down_channels[i+1], time_emb_dim ) for i in range(len(down_channels)-1 )])
    self.ups = nn.ModuleList( [ Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1 )] )
  
  def forward(self, x):
    pass




