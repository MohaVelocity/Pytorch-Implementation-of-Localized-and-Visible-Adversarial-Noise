# Pytorch-Implementation-of-Localized-and-Visible-Adversarial-Noise

All code in this repository is based on the following paper
 
Danny Karmon, Daniel Zoran, Yoav Goldberg, (2018). 
LaVAN: Localized and Visible Adversarial Noise. 

Link: https://doi.org/10.48550/arXiv.1801.02608

The repository includes code for functions that can be used to easily create  and evaluate adversarial patches for any pretrained model, along with a demo.

## Documentations For Functions

```cifar_make_patch(model,imgs,labels,epochs,epsilon,device,patch_size=5,target=5,clamp=True,clamp_range=[0.0,1.0],verbose=True)```

Trains a patch of arbirtary size on a group of trainig images.

**Parameters:**

    model: Pretrained model for training the patch. Input shape must be [1,3,32,32]

    imgs: List of images used for training. Must be a Python list of tensors with shape [1,3,32,32]

    labels: List of labels used for training. Must be a python list of tensors with shape [1]

    epochs: The number of iterations on all of the images and labels. Should be multiple of 10 for verbose.

    epsilon: The coefficient used on each iteration of back propagation

    device: The device to put all tensors. Ex 'cuda', 'cpu'.

    patch_size: The length of one dimension of the patch square.

    target: Target label number to tune the patch to.

    clamp: To enable the limit for the patch pixel values in the image domain.

    clamp_range: A list of 2 values with the first being the minimum and second being the maximum of image domain

    verbose: if True, shows the training process in 11 parts deviding the epochs by 10 and first epoch.

**return: a patch tensor of shape `[1,3,patch_size,patch_size]`**


```cifar_evaluate(model,imgs,labels,device,target,patch_small)```

Evaluates on the given images and outputs the target accuracy average, the 90% confidence accuracy average
and the proportion of outputs that were different from the correct source.

**Parameters:**

    model: pretrained model used for evaluation.

    imgs: List of images used for evaluation. Must be a Python list of tensors with shape [1,3,32,32]

    labels: List of labels used for evaluation. Must be a python list of tensors with shape [1]

    device: The device to put all tensors. Ex 'cuda', 'cpu'.

    target: Target label number to used for evaluation.

    patch_small: tensor of shape [1,3,patch size,patch size] to be used as the adverserial patch.

**return: None**


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
