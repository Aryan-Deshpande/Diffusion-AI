import torch
from PIL import Image
""""from UNet import UNetAE
#from CLIP import CLIPmodel

import torch
from torchvision import transforms
import clip
from PIL import Image

import torch
import torch.nn.functional as F

def linear_beta_scheduler(timestamps=None, start=0.0001, end=0.02):
    return torch.linspace(start, end, timestamps)

def get_index_from_list(vals, t, x_shape):
  batch_size = t.shape[0]
  out = vals.gather(-1, t)
  return out.reshape(batch, *((1,) * (len(x_shape) - 1 )))

def function_to_noise(x_0, t):
  noise = torch.rand_like(x_0)

  sqrt_alphas_cumprod_t =  get_index_from_list(sqrt_alphas_cumprod_t, t, x_0.shape)
  sqrt_one_minus_alphas_cumprod = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, x_0.shape)
  
  return (sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device))

t = linear_beta_scheduler(10)
print(linear_beta_scheduler(10))

#print(get_index_from_list(torch.tensor([10,12,3,4,15]), t, 512))
image = Image.open('./da.png')
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])
generated_image = trans(image)
a = function_to_noise(generated_image, t)

print(a)
"""

# Utilizing linear beta scheduler here
"""class Diffusion:
  def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size, device='cuda'): 
    self.noise_steps = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.img_size = img_size

    self.device = device

    self.beta = self.prepare_noise_schedule().to(device)
    self.alpha = 1 - self.beta

    # cumprod calculates the cumulative product till n
    # example: [1,2,3,4] -> [1,2,6,24]
    self.alpha_cum_prod = torch.cumprod(self.alpha, dim=0)

  def prepare_noise_schedule(self):
    return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

  def forward_diffusion(self, x_0, t):

    # we do this to make sure that the noise is the same for all the steps
    sqrt_alpha_hat = torch.sqrt(self.alpha_cum_prod[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_cum_prod[t])[:, None, None, None]
    s = torch.rand_like(x_0)

    return sqrt_alpha_hat * x * sqrt_one_minus_alpha_hat * s , s
  
  def sample_timesteps(self, n):
    return torch.randint(0, self.noise_steps, (n,))"""

# instead of adding sequentially
# noise can be added to all the timestamps at once, ...
# this is since sum of gaussian dist is gaussian 

from torchvision import transforms

def linear_beta_scheduler(beta_start, beta_end, noise_steps):
  a = torch.linspace(beta_start,beta_end, noise_steps)
  return a

b = 1e-4
# here the betas represent the amount of noise used
# it is also known as the scheduler
betas = linear_beta_scheduler(b, 0.02, 10)

# alphas represent the amount of information retained after noising
alphas = 1-betas
print(alphas) # [0.9999, 0.9977, 0.9955, 0.9933, 0.9911, 0.9888, ...

image = Image.open('./da.png')
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])
generated_image = trans(image)

revtrans = transforms.Compose([
    transforms.ToPILImage(),
])

# noise for single step
noise_image = torch.rand_like(generated_image)
image = revtrans(noise_image)
##image.show()

# cumulative alpha product
# allows to sample an image noised, at a particular timestep, all at once
alpha_cum_prod = torch.cumprod(alphas, dim=0)
print(alpha_cum_prod)

# mean & variance of gaussian distribution
# N(xt, sqrt(alpha), beta)
t = 1
sqrt_alpha_cum_prod = torch.sqrt(alpha_cum_prod[t])

sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_cum_prod[t])

rand = torch.rand_like(generated_image)

"""n,j = sqrt_alpha_cum_prod + sqrt_one_minus_alpha_hat * rand, rand

new = revtrans(n)
new.show()"""

class Diffusion:
  def __init__(self, beta_start, beta_end, noise_step):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.noise_step = noise_step
    self.beta = self.beta_schedular(self.beta_start, self.beta_end, self.noise_step)
    self.alpha = 1 - self.beta
    self.alpha_cum_prod = torch.cumprod(alpha_cum_prod, dim=0)

  def beta_schedular(self, beta_start, beta_end, noise_step):
    return torch.linspace(beta_start, beta_end, noise_step)
    
  def noise_image(self, x0, t):
    sqrt_alpha_cum_prod = torch.sqrt(alpha_cum_prod[t])
    sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - alpha_cum_prod[t])

    rand = torch.rand_like(x0)
    return ( sqrt_alpha_cum_prod + ( sqrt_one_minus_alpha_cum_prod * x0) ), x0

beta_start=0.0004
beta_end = 0.02
noise_step = 10
noiser = Diffusion(beta_start, beta_end, noise_step)

x0, n0 = noiser.noise_image(generated_image, 1)
x10, n10 = noiser.noise_image(generated_image, 4)

x0image = revtrans(x0)
x10image = revtrans(x10)

x0image.show()
x10image.show()