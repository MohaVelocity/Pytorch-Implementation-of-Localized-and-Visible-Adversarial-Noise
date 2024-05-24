# Welcome to Localized and Visible Adversarial Noise with Pytorch!


## Overview
All code in this repository is based on the following paper
[Danny Karmon, Daniel Zoran, Yoav Goldberg, (2018). LaVAN: Localized and Visible Adversarial Noise.](https://doi.org/10.48550/arXiv.1801.02608) 

This repository is a faithful recreation of the adversarial patch generation mechanism explained in the paper above using Pytorch 2.0.
The code was weitten as part of a research project at the [Hara-Lab](http://www.cad.ce.titech.ac.jp/) of [Tokyo Institute of Technology](https://www.titech.ac.jp/english).
The repository is mainly composed of the following  
* generate_patch: python file with options for patch generation using models pretrained on ImageNet-1K
* Demo_for_patch_creation: Ipython demo file for step by step guide useful for experimentation
* utils: python file with functions and classes for patch generation which can be modifeied for other applications



![Eagle with jelly fish patch](/adversarial_examples/eagle_to_jelly.png) ![Bear to Thimble patch](/adversarial_examples/bear_to_thimble.png)
![Samoyed to Green Mamba](/adversarial_examples/samoyed_to_mamba.png) ![Starfish to dome](/adversarial_examples/starfish_to_dome.png)

**Some example patches and results for the Inception_V3 model.**
* Eagle 4.7%, Jelly Fish 92.3%
* Brown Bear 11.2%, Thimble 78.8%
* Samoyed 2.3%, Green Mamba 94.7%

## Quick Start Guide

#### Prepare images for training
The generate_patch.py file can be used to generate an adversarial patch on a group of user defined images with a choice of assortment of pretrained models. To prepare images for patch generation, it is required to store images in a folder along with a labels.txt file that includes the name of the images to be used and their Imagenet-1K label as a number.
As an example here is how the default labels.txt looks like in the images folder.
```
bald_eagle.jpg,22
boa_constrictor.jpg,61
box_turtle.jpg,37
brown_bear.jpg,294
cock.jpg,7
samoyed.jpg,258
spotted_salamander.jpg,28
starfish.jpg,327
tiger.jpg,292
tiger_shark.jpg,3
```
Make sure to have one item on each row and separate them with a comma.

#### Options for generate_patch.py
Here are the options used for defining how the patch should be made.  
The default values are in the bracket.
```
--target: index number for the target output for the model
-i --input path: path to the directory where the images used for training are ('./images')
-o --output_path: 'path to the directory for the output patch to be stored' ('./')
--epsilon: epsilon value for training (0.05)
--clamp_val: clamp standardized patch to [-clamp_val,clamp_val] (2)
--epochs: epochs of training (30)
--iter: gradient descent steps for each image in each epoch (100)
--device: what device is used for training, default is cuda ('cuda')
--patch_size: size of the patch in pixels as a single integer, the patch is a square (50)
--model_name: the model to attack (inception_v3)
```
The model_name option can accept the following model names.
inception_v3, resnet50, resnext50, mobilenet_v3_large, swin_b

Both the demo Ipython file and generate_patch.py are setup by default to work with the 10 images in the images folder and generate a 50x50 patch for the Inception_V3 model.


## Adaptable functions in utils

The utils file includes two functions, manual_single_epoch and apply_evaluate_patch which can be used to train and evaluate adversarial patches for various models with various datasets beyond just Imagenet-1K
Below are the explanation of what they require as input and what they return.

```manual_single_epoch(model,img,label,epsilon,device,patch_small,patch_corner,target=5,clamp=True,clamp_range=[0.0,1.0])```

Does a single back propagation on an input patch on a given place on a given image with a given model.

**Parameters:**

    model: Pretrained model for training the patch. Input shape must be [1,3,M,N]

    img: image used for training, Must be tensor with shape [1,3,M,N]

    labels: List of labels used for training. Must be a python list of tensors with shape [1].

    epsilon: The coefficient used on each iteration of back propagation

    device: The device to put all tensors. Ex 'cuda', 'cpu'.

    patch_small: The adverserial patch. Must be tensor with shape [1,3,X,Y]

    patch_corner: The coordinates of top left of where patch should be applied to. must be list with length 2.

    target: Target label number to tune the patch to.

    clamp: To enable the limit for the patch pixel values in the image domain.

    clamp_range: A list of 2 values with the first being the minimum and second being the maximum of image domain

**return: a patch tensor of shape `[1,3,patch_size,patch_size]`**



    

```apply_evaluate_patch(patch_small,model,img,patch_corner)```

Applies an adversarial patch to a given image and outputs the probabilites of all classes.

**Parameters:**

    patch_small: The adverserial patch. Must be tensor with shape [1,3,X,Y]

    model: Pretrained model for testing the patch. Input shape must be [1,3,M,N]

    img: Image to apply the adversarial patch to, Must be tensor with shape [1,3,M,N]

    patch_corner: The coordinates of top left of where patch should be applied to. must be list with length 2.

**return: a one dimentional tensor with length of output classes**
