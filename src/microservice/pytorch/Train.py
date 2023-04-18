from UNet import UNetAE
#from CLIP import CLIPmodel

import torch
from torchvision import transforms
import clip
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Training():
    for i in range(100):

        optimizer.zero_grad()

        print('Before forward noising')
        x_noisy_image, noise = forward_diff()
        print('before unet forward pass')
        noise_predicted = UNet(x_noisy_image)

        # finding the CLIP loss
        print('before computing clip loss')
        clip_loss = computer_clip_loss()

        print('before computing l1 loss')
        l1_loss = torch.nn.functional.l1_loss(noise_pred, noise)

        # computing the final loss, utilizing the CLIP loss for guidance
        final_loss = l1_loss + clip_loss
        print('calculating final loss')

        final_loss.backward()
        optimizer.step()

        print('..........')

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

#image = Image.open('./da.png')
image = Image.open('./das.jpg')
trans = transforms.Compose([
    transforms.ToTensor(),

])
generated_image = trans(image)
print(generated_image.shape, ' old shape ')
generated_image = generated_image.view(1, 3, 142,186)
print(generated_image.shape, ' new shape ')

generated_image = generated_image.to(device)

generated_image = torch.nn.functional.interpolate(generated_image, size=(224, 224), mode='bilinear', align_corners=False)

compute_clip_loss(generated_image, "city skyline with lava on the road")

# Total number of parameters 
#print( sum(i.numel() for i in model.parameters() ))