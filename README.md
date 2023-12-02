# Pytorch-Implementation-of-Localized-and-Visible-Adversarial-Noise

All code in this repository is based on the following paper
 
Danny Karmon, Daniel Zoran, Yoav Goldberg, (2018). 
LaVAN: Localized and Visible Adversarial Noise. 

Link: https://doi.org/10.48550/arXiv.1801.02608

The repository includes code for functions that can be used to easily create  and evaluate adversarial patches for any pretrained model, along with a demo.

## Documentations For Functions

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
