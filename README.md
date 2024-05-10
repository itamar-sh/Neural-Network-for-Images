# Neural-Network-for-Images
CNN, AutoEncoder, Generative Models, Score Matching, StyleGan, Self Supervision.<br/>
For all those projects I used pytorch, tensorflow and matplotlib.


## CNNs
Classification using convolutional neural networks.<br/>
Working on CIFAR10 dataset, track results using [wandb](https://docs.wandb.ai/guides).<br/>
Exploring different architectures, understaing importance of non-linearity and cascaded receptive field.<br/>

Example of variance-bias tradeoff:<br/>
<img
  src="resources/images/CNNs exmpale.png"
  title="CNN example"
  style="display: inline-block; margin: 0 auto;" width="350" height="200"><br/>


## AutoEncoder with Conv layers
Auto-Encoding (AE) and transfer-learning with the AE over the MNIST digits dataset.<br/>
We tested aspects of AE such as dimensionality reduction, Interpolation, Decorrelation.<br/>
I used BCELoss as criterion and Adam for the optimizer.<br/>
The encoder is Conv layers with sigmoid and FC layer in the end, The decoder has one FC and Deconv layers.

Demonstration of reconstruction success in relation to latent space size:<br/>
<img
  src="resources/AutoEncoder/latent space test.png"
  title="AE example"
  style="display: inline-block; margin: 0 auto;" width="350" height="200"><br/>


## Generative Adversarial Networks (GANs)
In this part I trained a generative adversarial model to produce images of a particular distribution.<br/>
generator network can follow the DCGAN architecture. Generator: FC with convs. Discrimator: classification network which decide: real or fake image..<br/>

Main points: Loss saturation, Model Inversion and Image restoration including denoise and inpainting.<br/>

Example of denoising using Gan and latent optimization:<br/>
<img
  src="resources/images/Denoise Example using Gan.png"
  title="Gan example"
  style="display: inline-block; margin: 0 auto;" width="350" height="200"><br/>


