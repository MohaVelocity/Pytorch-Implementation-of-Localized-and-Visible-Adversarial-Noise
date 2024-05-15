import numpy as np
import os
import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

def args_check(input_path,output_path,epsilon,iter_num,target,epochs,patch_size,model_name):
    if not os.path.exists(input_path):
        raise Exception('The provided input path does not exist')
    if not os.path.exists(output_path):
        raise Exception('The provided output path does not exist')
    if epsilon <= 0:
        raise Exception('Epsilon must be greater than 0')
    if iter_num <= 0:
        raise Exception('Number of iterations per image per epoch must be greater than 0')
    if target == None:
        raise Exception('Target index is not provided')
    if not (0 <= target <= 999):
        raise Exception('Target index must be between 0 and 999')
    if epochs <= 0:
        raise Exception('Epoch must be larger than 0')
    if patch_size <= 0:
        raise Exception('Patch size must be greater than 0')
    
    model_names = ['inception_v3','resnet50','resnext50','mobilenet_v3_large','swin_b']

    if model_name not in model_names:
        raise Exception('The input model is not recognized')
    

#device = (torch.device('cuda') if torch.cuda.is_available()
#          else torch.device('cpu'))

def manual_single_epoch(model,img,label,epsilon,device,patch_small,patch_corner,target,clamp=True,clamp_range=[0.0,1.0]):

    label_list = [label]
    label_t = torch.tensor(label_list).to(device=device)

    patch_size = patch_small.shape[2]
    h_bottom = patch_corner[0]
    h_top = h_bottom + patch_size
    w_left = patch_corner[0]
    w_right = w_left + patch_size
    
    patch = torch.zeros(img.shape)
    mask = torch.zeros(img.shape)
    mask[:, :, h_bottom: h_top , w_left:w_right] = 1
    patch[:, : , h_bottom:h_top , w_left:w_right ] = patch_small
    
    patch = patch.to(device=device)
    mask = mask.to(device=device)
    
    adv_x = torch.mul((1-mask),img) + torch.mul(mask,patch)
    
    target_list = [target]
    target_t = torch.tensor(target_list).to(device=device)

    adv_x = Variable(adv_x.data, requires_grad=True)
    output = model(adv_x)
            
    loss = F.cross_entropy(output, target_t) - F.cross_entropy(output, label_t)
    loss.backward()

    adv_grad = adv_x.grad.clone()
    adv_x.grad.data.zero_()

    patch -= adv_grad * epsilon

    if clamp:
        patch = torch.clamp(patch, clamp_range[0], clamp_range[1])
                
    return loss.item(), patch[:,:,h_bottom:h_bottom+patch_size,w_left:w_left+patch_size]


def apply_evaluate_patch(patch_small,model,img,patch_corner,device):
    
    patch_size = patch_small.shape[2]  
    x = img.to(device=device)
    x.requires_grad = False
        
    h_bottom = patch_corner[0]
    h_top = h_bottom + patch_size
    w_left = patch_corner[0]
    w_right = w_left + patch_size
    
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

def read_labels(path):
    with open(path) as file:
        lines = file.readlines()
    image_names = []
    image_labels = []
    for line in lines:
        name_and_label = line.strip().split(',')
        image_names.append(name_and_label[0])
        image_labels.append(int(name_and_label[1]))
    return image_names,image_labels


def load_and_preprocess_images(path,device,preprocess):
    images = []
    image_labels = []
    if os.path.exists(os.path.join(path,'labels.txt')):
        image_names,image_labels = read_labels(os.path.join(path,'labels.txt'))
        images = [Image.open(os.path.join(path,image_name)) for image_name in image_names]
    else:
        print("No labels.txt located, resuming by assuming the original predicted class is correct")
        image_paths = glob.glob(os.path.join(os.getcwd(),path,'*'))
        images = [Image.open(image_path) for image_path in image_paths]

    crop = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

    preprocessed_images = [preprocess(image).unsqueeze(0).to(device=device) for image in images]
    cropped_images = [crop(image) for image in images]

    return cropped_images,preprocessed_images,image_labels

def load_model(model_name,device):
    if model_name == 'inception_v3':
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights).to(device=device)
        
    if model_name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights).to(device=device)
        
    if model_name == 'resnext50':
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
        weights = ResNeXt50_32X4D_Weights.DEFAULT
        model = resnext50_32x4d(weights=weights).to(device=device)

    if model_name == 'mobilenet_v3_large':
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights).to(device=device)

    if model_name == 'swin_b':
        from torchvision.models import swin_b, Swin_B_Weights
        weights = Swin_B_Weights.DEFAULT
        model = swin_b(weights=weights).to(device=device)

    preprocess = weights.transforms()
    model.eval()
    return model,preprocess

def print_nonadversarial_result(preprocessed_images,model,label_list):
    assumed_label_list = []
    for i,image in enumerate(preprocessed_images):
        with torch.no_grad():
            output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        if label_list:
            print("Image {} confidence for correct class: {:.3f}".format(i+1,probabilities[label_list[i]].item()))
        else:
            print("Image {} highest confidence: {:.3f}, label index: {}".format(i+1,probabilities.max().item(),probabilities.argmax().item()))
            assumed_label_list.append(probabilities.argmax().item())
    if assumed_label_list:
        label_list = assumed_label_list

def generate_patch(model,preprocessed_images,label_list,target,epsilon,epochs,iter_num,patch_size,device,clamp_val):
    patch_small = np.random.uniform(0.0, 1.0, (1, 3, patch_size, patch_size))
    patch_small = torch.from_numpy(patch_small)
    clamp_range = [-clamp_val,clamp_val]
    for i in range(epochs):
        loss_cum = 0
        for image,label in zip(preprocessed_images,label_list):
            for _ in range(iter_num):
                xcorner = np.random.randint(0,image.shape[2]-patch_size-1)
                ycorner = np.random.randint(0,image.shape[2]-patch_size-1)
                corner_cords = [xcorner,ycorner]
                loss, patch_small = manual_single_epoch(model,image,label,epsilon,device,patch_small,\
                corner_cords,target,clamp=True,clamp_range=clamp_range)
                loss_cum += loss
            
        if (i+1)%(epochs//10) == 0: 
            avg_loss = loss_cum/(len(preprocessed_images)*iter_num)
            print("Average loss at epoch {}: {:04f}".format(i+1,avg_loss))
    return patch_small

def print_numeric_results(model,preprocessed_images,label_list,patch,target,device):
    soft_fool = 0
    hard_fool = 0
    patch_size = patch.shape[2]
    for i,(image,label) in enumerate(zip(preprocessed_images,label_list)):
        xcorner = np.random.randint(0,image.shape[2]-patch_size-1)
        ycorner = np.random.randint(0,image.shape[2]-patch_size-1)
        probabilities = apply_evaluate_patch(patch,model,image,[xcorner,ycorner],device)
        original_label_probability = probabilities[label].item()
        target_label_probability = probabilities[target].item()
        if target_label_probability >= original_label_probability:
            soft_fool += 1
        if target_label_probability >= 0.9:
            hard_fool += 1
        print("Image {} original label confidence: {:.3f}, target label confidence: {:.3f}"\
              .format(i+1,original_label_probability,target_label_probability))
    print("Target has a larger probability in {}/{} cases".format(soft_fool,len(preprocessed_images)))
    print("Target has a probability greater than 0.9 in {}/{} cases".format(hard_fool,len(preprocessed_images)))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor