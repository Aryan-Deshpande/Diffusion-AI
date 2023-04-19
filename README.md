# Latent Diffusion Multi-Model with CLIP guidance

## This is a ML web application that enables users to enter text prompts of choice, of which in-accordance an Image is generated using multiple deep learning models



## Run it Locally
```sh
git clone https://github.com/Aryan-Deshpande/Latent-Diffusion-AI
docker compose up
```
<sp>then go to localhost:3001<sp>

# Diffusion Working
- Essentially the idea is such that we use a Gaussian distribution to noise the image, at particular timesteps.
- Instead of sequentially utilizing the output of the previous timestep to apply noise until t timestamp
- we directly sample the noised image at all timestamps Xt, this can be done because the sum of Gaussian distribution is nothing but Gaussian 
- 