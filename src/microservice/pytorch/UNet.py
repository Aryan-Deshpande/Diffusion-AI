import torch 
import torch.nn as nn

from torchvision import transforms

from Positional_Embeddings import SinosudialPositionalEncoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Block(nn.Module):
  def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super(Block, self).__init__()
    
    self.time_mlp = nn.Linear(time_emb_dim, out_ch, device=device) # map the time embedding 
                                                    # to the out embed

    #self.example = nn.Conv2d(64, 128, 3, padding=1, device=device)

    if up == True:
      self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, device=device)
      self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1, device=device)
    else:
      self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, device=device)
      self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1, device=device)

    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, device=device)
    self.pool = nn.MaxPool2d(3, stride=2)
    self.bnorm = nn.BatchNorm2d(out_ch, device=device)
    self.relu = nn.ReLU()
  
  def forward(self, x, t):
    # first conv layer
    print(('x shape', x.shape))
    #print('block first conv layer')
    print('convolution layer weights, biases', self.conv1.weight.shape, self.conv1.bias.shape)

    h = self.bnorm(self.relu(self.conv1(x) )) 
    print('h shape', h.shape)

    # time embedding
    time_emb = self.relu(self.time_mlp(t))
    print('time_emb')

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

class UNetAE(nn.Module):
  def __init__(self):
    super(UNetAE, self).__init__()

    image_channels = 3
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (1024, 512, 256, 128, 64)

    out_dim = 1
    time_emb_dim = 32

    # Time embedding
    self.time_mlp = nn.Sequential(
        SinosudialPositionalEncoder(time_emb_dim),
    )

    self.time_mlp_linear = nn.Sequential(
      nn.Linear(time_emb_dim, time_emb_dim),
      nn.ReLU()
    )

    self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1, device=device)

    self.downs = nn.ModuleList( [ Block( down_channels[i], down_channels[i+1], time_emb_dim ) for i in range( len(down_channels)-1 ) ] )
    self.ups = nn.ModuleList( [ Block( up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range( len(up_channels)-1 ) ] )
    
    self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

  def forward(self, x, t):
    #print('In the UNet module', x)

    t = self.time_mlp(t) # utilized to embed the time
    #x = self.time_mlp_linear(x)
    #print(t, 'timemlp')
    #print(t.shape, ' shape of timemlp')
    
    #print(self.conv0.weight.get_device(), self.conv0.bias.get_device(), 'shapes of parameters')
    first_conv_out = self.conv0(x)
    #print('after first conv', first_conv_out.shape)

    x = first_conv_out
    print(x.shape, ' x.shape')

    residual = []

    print(self.downs)
    print(self.ups)
    for down in self.downs:

      x = down(x, t)

      residual.append(x)

    # residual connection ?
    # attention block
    # residual connection ?
    
    for up in self.ups:
      residual_x = residual.pop()

      # add the residual as a additional channel
        
      x = torch.cat( (residual_x, x), dim=1)
      x = up(x, t)

    print(x.shape, 'out of UNet module')

    return self.output(x)

"""def loss(model, x_0, t):
  x_noisy, noise = forward_diff()
  noise_pred = unet(x)

  clip_loss = compute_clip_loss()

  l1_loss = torch.nn.functional.l1_loss(noise_pred, noise)

  final_loss = l1_loss + clip_loss

  return final_loss"""




