# See360 (Novel Panoramic View interpolation and rendering)

By Zhi-Song Liu, Marie-Paule Cani and Wan-Chi Siu

This repo only provides simple testing codes, pretrained models and the network strategy demo.

We present See360, which is a versatile and efficient framework for 360-degree panoramic view interpolation using latent space viewpoint estimation.

Please check our [paper]([https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9709215](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9709215))

# BibTex


        @InProceedings{Liu2022see360,
            author = {Liu, Zhi-Song and Cani, Marie-Paule and Siu Wan-Chi},
            title = {See360: Novel Panoramic View interpolation},
            booktitle = {IEEE Transactions on Image Processing},
            year = {2022},
            pages={1857-1869},
            doi={10.1109/TIP.2022.3148819}
        }
  
# Demo
We show four examples of real-world and virtual-world view rendering.

![eg1](/figures/hunghom_our.gif)  ![eg2](/figures/lab_our.gif)
![eg3](/figures/archinterior_our.gif)  ![eg4](/figures/urbancity_our.gif) 

# For proposed See360 model, we claim the following points:

• To generate high-quality, photo-realistic images without requiring 3D information, we propose a Multi-Scale Affine Transformer (MSAT) to render reference views in the feature domain using 2D affine transformation. Instead of learning one-shot affine transform to reference views, we learn multiple affine transforms in a coarseto-fine manner to match the reference features for view synthesis.
• Furthermore, to allow users to interactively manipulate views at any angle, we introduce the Conditional Latent space AutoEncoder (C-LAE). It consists of 1) patch based correlation coefficients estimation and 2) conditional angle encoding. The former enables finding global features for 3D scene coding and the latter introduces target angles as one-hot binary codes for view interpolation.
• In addition, we provide two different types of datasets to train and test our model. One is the synthetic 360 images collected from the virtual world, including UrbanCity360
and Archinterior360. Another is the real 360 images collected from real-world indoor and outdoor scenes, including HungHom360 and Lab360. Semantic segmentation maps are also provided for all datasets. Our tests in the wild show that See360 can also be used, to some extent, with unknown real scenes. With a small number of training images (about 24 images) required, it takes 10 mins training to reconstruct the 360 view rendering.

# Dependencies
    Python > 3.0
    OpenCV library
    Pytorch > 1.0
    NVIDIA GPU + CUDA

# Complete Architecture
The complete architecture is shown as follows,

![network](/figures/network.png)
