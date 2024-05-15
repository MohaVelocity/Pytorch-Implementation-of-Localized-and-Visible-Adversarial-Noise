import numpy as np
import datetime
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ipywidgets import interact
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import torch.nn.functional as F

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

def manual_single_epoch(model,img,label,epsilon,device,patch_small,patch_corner,target,clamp=True,clamp_range=[0.0,1.0]):
    
    revert_train = False
    if model.training:
        revert_train = True
        model.eval()
        
    patch_size = patch_small.shape[2]
    
    if patch_corner[0] + patch_size > img.shape[2] or patch_corner[1] + patch_size > img.shape[2]:
        raise Exception("Patch exceeds the borders of the image")
    
    model = model.to(device=device)
    
    x = img.to(device=device)
    x.requires_grad = False

    label_list = [label]
    label_t = torch.tensor(label_list).to(device=device)
        
    h_bottom = patch_corner[0]
    h_top = h_bottom + patch_size
    
    w_left = patch_corner[0]
    w_right = w_left + patch_size
    
    n, c, h, w = x.shape
    
    patch = torch.zeros(x.shape)
    mask = torch.zeros(x.shape)
        
    mask[:, :, h_bottom: h_top , w_left:w_right] = 1
    
    patch[:, : , h_bottom:h_top , w_left:w_right ] = patch_small
    
    patch = patch.to(device=device)
    mask = mask.to(device=device)
    
    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    
    target_list = [target]
    target_t = torch.tensor(target_list).to(device=device)


    adv_x.requires_grad = True

    output = model(adv_x)
            
            
    loss = F.cross_entropy(output, target_t) - F.cross_entropy(output, label_t)
    loss.backward()

    adv_grad = adv_x.grad.clone()
    adv_x.grad.data.zero_()

    patch -= adv_grad * epsilon


    if clamp:
        #adv_x = torch.clamp(adv_x, clamp_range[0], clamp_range[1])
        patch = torch.clamp(patch, clamp_range[0], clamp_range[1])
                
                
    if revert_train:
        model.train()
    
    return loss.item(), patch[:,:,h_bottom:h_bottom+patch_size,w_left:w_left +patch_size]


def apply_evaluate_patch(patch_small,model,img,patch_corner):

    revert_train = False
    if model.training:
        revert_train = True
        model.eval()
        
    patch_size = patch_small.shape[2]
    
    if patch_corner[0] + patch_size > img.shape[2] or patch_corner[1] + patch_size > img.shape[2]:
        raise Exception("Patch exceeds the borders of the image")
    
    model = model.to(device=device)
    
    x = img.to(device=device)
    x.requires_grad = False
        
    h_bottom = patch_corner[0]
    h_top = h_bottom + patch_size
    
    w_left = patch_corner[0]
    w_right = w_left + patch_size
    
    n, c, h, w = x.shape
    
    patch = torch.zeros(x.shape)
    mask = torch.zeros(x.shape)
        
    mask[:, :, h_bottom: h_top , w_left:w_right] = 1
    
    patch[:, : , h_bottom:h_top , w_left:w_right ] = patch_small
    
    patch = patch.to(device=device)
    mask = mask.to(device=device)
    
    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)

    adv_x.requires_grad = False

    output = model(adv_x)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities