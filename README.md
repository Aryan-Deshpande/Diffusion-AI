### Needs fixes

# Latent Diffusion Multi-Model with CLIP guidance
This is a Web Application that pertains to "generating" Images based on the specific prompt.
A machine learning model is implemented, which utilizes the input prompt as a guidance to the kind of image that should be denoised.
An iterative process of Denoising a completely obscure image, until termination of the loop.
Utilizes multiple Deep Learning Model Arcitectures
- *Attention Modules* [https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf]
- *CLIP* ( Contrastive Language Image Pre-Training ) Model [https://paperswithcode.com/method/clip]
-  *UNet* [https://paperswithcode.com/method/u-net]

# During Training
*1)* The idea is such that we use a Gaussian distribution to noise the training image at particular timestamps.
Instead of sequentially utilizing the output of the previous timestep to apply noise until t timestamp, we directly sample the noised image at all timestamps Xt, this can be done because the sum of Gaussian distribution is nothing but Gaussian itself.
*2)* Consequently, the output of this process is then fed into a UNet Model along with the text label for the pertaining image.
The goal of the UNet model is to transform the text label, and the image into a smaller dimensional space, famously known as the latent space.
The latent space is the representation of compressed data, containing data that are similar, are closer together. ( Represents the Probability Distribution of the data )
*3)* Contrastive Loss and Cosine Similarity are utilized as guidance to optimize the generation of these images. The weights in the attention modules are shifted accordingly, until the loss reaches a minimal amount.

## Run it Locally
```sh
git clone https://github.com/Aryan-Deshpande/Latent-Diffusion-AI
docker compose up
```
<sp> --> then go to localhost:3001<sp>

# Diffusion Working

Algorithm

- Essentially the idea is such that we use a Gaussian distribution to noise the image, at particular timesteps.
- Instead of sequentially utilizing the output of the previous timestep to apply noise until t timestamp
- we directly sample the noised image at all timestamps Xt, this can be done because the sum of Gaussian distribution is nothing but Gaussian

### Papers Referenced / Used
- **Attention is All you Need** [https://arxiv.org/abs/1706.03762]
- **Contrastive Language Image Pre-Training** [https://browse.arxiv.org/pdf/2103.00020.pdf]
- **U-Net: Convolutional Networks for Biomedical Image Segmentation** [https://arxiv.org/abs/1505.04597]
- **High-Resolution Image Synthesis with Latent Diffusion Models**[https://arxiv.org/abs/2112.10752]

