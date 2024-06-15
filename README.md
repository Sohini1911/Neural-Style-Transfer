# Neural-Style-Transfer

## 1. Introduction
Given a pair of images, the process of combining the “style”
of one image with the “content” of the other image to create
a piece of synthetic artwork is known as style transfer. Style
transfer is a popular topic in both academia and industry, due
to its inherent attraction and wide applicability. [Gatys et al.
(2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
demonstrated a generalized style transfer technique
by exploiting feature responses from a pre-trained CNN,
opening up the field of neural style transfer, or the process
of using neural networks to render a given content image
in different artistic styles. Since then, many improvements,
alternatives, and extensions to the algorithm of Gatys et al.
has been proposed.

This work is an implementation of the paper [Image Style Transfer Using Convolutional Neural Networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

## 2.Dataset and Features
We are using the pre-trained model [SqueezeNet](https://arxiv.org/abs/1602.07360) model on [ImageNet](https://www.image-net.org/) dataset, specifically the version 'squeezenet1_1' available in the 'torchvision.models' module. SqueezeNet is a smaller, efficient convolutional neural network designed to achieve similar accuracy to AlexNet with fewer parameters, making it suitable for tasks where computational efficiency and memory usage are important.

In artwork generation and algorithm evaluation, we use a
variety of content and style images. As content images, we
use personal photos. As style images, we use well-known artworks
of Van Gogh, Leonardo da Vinci, Edvard Munch, etc., all of which are in the public domain.


## 3. Method
Passing an image $\overset{\rightarrow}{x}$ through the CNN produces feature maps
{ $F^l (\overset{\rightarrow}{x})$ }. The style representation of the image at layer l
is represented by the Gram matrix:
$\mathcal{G}\left(F^{[l]}(\vec{x})\right)=\left[F^{[l]}(\vec{x})\right]\left[F^{[l]}(\vec{x})\right]^T$

To transfer the style of an artwork $\overset{\rightarrow}{a}$ onto a photograph $\overset{\rightarrow}{p}$
and produce a synthesized image $\overset{\rightarrow}{x}$, we initialize a random
image $\overset{\rightarrow}{x}$, and minimize the total loss function (Eq. 1) defined
as a linear combination of content loss (Eq. 2), style loss
(Eq. 3) and total variational loss (Eq. 4). In this way, the
output image $\overset{\rightarrow}{x}$ strives to match $\overset{\rightarrow}{p}$ in content and match $\overset{\rightarrow}{a}$
in style. Gatys’ paper only specified content loss and style
loss, but total variation regularization (Eq. 4) is a standard
addition to Gatys’ method, and helps reduce the random
noise in the output image $\overset{\rightarrow}{x}$.

$L{\text{total }} (\vec{p}, \vec{a}, \vec{x}) =\alpha L_{\mathrm{c}}(\vec{p}, \vec{x}) +\beta L_{\mathrm{s}}(\vec{a}, \vec{x})+\gamma L_{\mathrm{v}}(\vec{x})$  ...(1)

$L_{\mathrm{c}}=\sum_{l \in \mathcal{C}}w_{\mathrm{c}}^{[l]}\left\||F^{[l]}(\vec{p})-F^{[l]}(\vec{x})\right\||_2^2$ ...(2)

$L_s^\ell =  \sum_{l \in \mathcal{S}}w_{\mathrm{s}}^\ell\left||G_{\mathrm{s}}^\ell - A_{\mathrm{s}}^\ell\right||_2^2$  ...(3)

$L_{\mathrm{v}}=w_{\mathrm{t}}\sum_{i, j}\left(\left|x_{(i, j)}-x_{(i+1, j)}\right|+\left|x_{(i, j)}-x_{(i, j+1)}\right|\right)$ ...(4)

In equations above, $\mathrm{C}$ is the set of content representation
layers, $\mathrm{S}$ is the set of style representation layers, the weights $w_{\mathrm{c}}^{[l]}$, $w_{\mathrm{s}}^{[l]}$ and $w_{\mathrm{t}}$
determine the relative weights between loss obtained at multiple style and content representation
layers in the CNN. The weights α, β, γ determine the relative weights between content, style, and total variational
loss. All these parameters are hyperparameters that need to
be manually chosen by the user to obtain the most visually
pleasing result.

### This figure have been taken from Gatys’ paper that demonstrates the style transfer algorithm:-
![image](https://github.com/Sohini1911/Neural-Style-Transfer/assets/134104045/8c0a1dd1-0484-4afa-b88f-7851427eebc6)
 First content and style features are extracted and stored. The style image $\overset{\rightarrow}{a}$ is passed through the network
 and it's style representation $A^l$ on all layers included are computed and stored(left). The content image $\overset{\rightarrow}{p}$ is passed through the network
 and the content representation $P^l$ in one layer is stored(right). Then a random white-noise image $\overset{\rightarrow}{x}$ is passed through the network and its
 style features $G^l$ and content features $F^l$ are computed. On each layer included in the style representation, the element-wise mean squared
 difference between $G^l$ and $A^l$ is computed to give the style loss $L_{style} $ (left). Also, the mean squared difference between $F^l$ and $P^l$ is
 computed to give the content loss $L_{content}$ (right). The total loss $L_{total}$ is then a linear combination between the content and the style loss.
 Its derivative with respect to the pixel values can be computed using error back-propagation(middle). This gradient is used to iteratively
 update the image $\overset{\rightarrow}{x}$ until it simultaneously matches the style features of the style image $\overset{\rightarrow}{a}$ and
 the content features of the content image $\overset{\rightarrow}{p}$
 (middle, bottom).

## 4. Results

### For example 1
| Hyperparameters | Values|
| --- | --- |
| $\mathrm{C}$ | Layer 3 |
| $\mathrm{S}$ | Layers 1,4,6,7 |
| $w_{\mathrm{c}}^{[l]}$ | 6e-2 |
| $w_{\mathrm{s}}^{[l]}$ | [300000, 1000, 15, 3] |
| $w_{\mathrm{t}}$| 2e-2 |
|(α, β, γ) | (1, 2, 1e-3)|
|learning rate| start at lr =3, at iteration 180 to lr = 0.1|

![image](https://github.com/Sohini1911/Neural-Style-Transfer/assets/134104045/1bbef331-a8ca-442f-88eb-c6b46ce712aa)

Iteration 0:
![image](https://github.com/Sohini1911/Neural-Style-Transfer/assets/134104045/51c65945-6a71-4ae3-bee2-4cc9017bcd9e)

Iteration 199: 
![image](https://github.com/Sohini1911/Neural-Style-Transfer/assets/134104045/ece60898-758e-4aae-b77d-5f5078a05eb8)

### For example 2
| Hyperparameters | Values|
| --- | --- |
| $\mathrm{C}$ | Layer 3 |
| $\mathrm{S}$ | Layers 1,4,6,7 |
| $w_{\mathrm{c}}^{[l]}$ | 3e-2 |
| $w_{\mathrm{s}}^{[l]}$ | [200000, 800, 12, 1] |
| $w_{\mathrm{t}}$| 2e-2 |
|(α, β, γ) | (1, 2, 1e-3)|
|learning rate| start at lr =3, at iteration 180 to lr = 0.1|

![image](https://github.com/Sohini1911/Neural-Style-Transfer/assets/134104045/23d68568-0053-4dec-83c7-c14f120c272a)

Iteration 199: 
![image](https://github.com/Sohini1911/Neural-Style-Transfer/assets/134104045/05bfea2f-b4aa-4096-b158-3ea33352436d)

More examples are shown in the notebook
