#from Diffusion import forward_diff
from UNet import UNetAE
#from CLIP import CLIPmodel

from Diffusion import DiffusionSched
from torch import Tensor
import torch
from torchvision import transforms

import clip
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Training(generated_image):
    # hardcoded work , needs to be dynamic
    beta_start=0.0004
    beta_end = 0.02
    noise_step = 10
    noiser = DiffusionSched(beta_start, beta_end, noise_step)

    # Initialize the Autoencoder Model
    model = UNetAE()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=1e-4)

    for i in range(noise_step):
        random_timestep = noiser.sample_n(1000).to(device)
        print('it is on device', random_timestep.get_device())

        print('Before forward noising')
        x_noisy_image, noise = noiser.noise_image(generated_image, i)

        print('before unet forward pass', x_noisy_image.shape, 'noisy image')
        noise_predicted = model(x_noisy_image, random_timestep)

        # finding the CLIP loss
        print('before computing clip loss')
        clip_loss = computer_clip_loss(generated_image, "city skyline with lava on the road")

        print('before computing l1 loss')
        l1_loss = torch.nn.functional.l1_loss(noise_pred, noise)

        # computing the final loss, utilizing the CLIP loss for guidance
        final_loss = l1_loss + clip_loss
        print('calculating final loss')

        #################################################################################
        final_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('..........')

        # so where does denoising happen ?

def compute_clip_loss(generated_image, text_prompt_encoded, clip_guidance_scale=0.1):
    with torch.no_grad():
        model, preprocess = clip.load("ViT-B/32")
        print('clip model loaded from clip library')

        for name, param in model.named_parameters(): # the CLIP model model.parameters are on the GPU tensors
            print(name, param.device)
        image_encoded = model.encode_image(generated_image)
        print(image_encoded, "image encoded")

        # tokenize the word
        text_tokenize = clip.tokenize([text_prompt_encoded]).to(device)

        text_encoded = model.encode_text(text_tokenize)

        print(text_encoded, "text encoded")
        # not exactly this, but something like this
        clip_loss = torch.nn.functional.cosine_similarity(image_encoded, text_encoded)
        print(clip_loss, "clip loss")

        clip_loss = -clip_loss * clip_guidance_scale
        print(clip_loss, "extended clip loss w/ clip guidance")

        return clip_loss

image = Image.open('./das.jpg')
trans = transforms.Compose([
    transforms.ToTensor(),

])
generated_image = trans(image)
print(generated_image.shape, ' old shape ')
generated_image = generated_image.view(1, 3, 142,186)
print(generated_image.shape, ' new shape ')

generated_image = generated_image.to(device)

generated_image = torch.nn.functional.interpolate(generated_image, size=(3, 3), mode='bilinear', align_corners=False)
print(generated_image.shape, 'shape of image before training function')
#compute_clip_loss(generated_image, "city skyline with lava on the road")

Training(generated_image)

# Total number of parameters 
#print( sum(i.numel() for i in model.parameters() ))