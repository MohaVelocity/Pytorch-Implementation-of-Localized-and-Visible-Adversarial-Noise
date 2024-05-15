from argparse import ArgumentParser
from utils import *
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('-i','--input_path',default='./images',help='path to the directory where the images used for training are')
    parser.add_argument('-o','--output_path',default='./',help='path to the directory for the output patch to be stored')
    parser.add_argument('--target',type=int,help='index number for the target output for the model')
    parser.add_argument('--epsilon',type=float,default=0.05,help='epsilon value for training')
    parser.add_argument('--clamp_val',type=float,default=2,help='clamp standardized patch to [-clamp_val,clamp_val]')
    parser.add_argument('--epochs',type=int,default=30,help='epochs of training')
    parser.add_argument('--iter',type=int,default=100,help='gradient descent steps for each image in each epoch')
    parser.add_argument('--device',default='cuda',help='what device is used for training, default is cuda')
    parser.add_argument('--patch_size',type=int,default=50,help='size of the patch in pixels as a single integer, the patch is a square')
    parser.add_argument('--model_name',default='inception_v3',help='the model to attack, see readme.md for options')
    args = parser.parse_args() 
    input_path = args.input_path
    output_path = args.output_path
    clamp_val = args.clamp_val
    epsilon = args.epsilon
    target = args.target
    epochs = args.epochs
    iter_num = args.iter
    device = args.device
    patch_size = args.patch_size
    model_name = args.model_name
    args_check(input_path,output_path,epsilon,iter_num,target,epochs,patch_size,model_name)

    device = (torch.device('cuda') if device=='cuda' else torch.device('cpu'))
    print(f"Training on device {device}.")

    model,preprocess = load_model(model_name,device)
    cropped_images,preprocessed_images,label_list = load_and_preprocess_images(input_path,device,preprocess)
    print_nonadversarial_result(preprocessed_images,model,label_list)

    patch = generate_patch(model,preprocessed_images,label_list,target,epsilon,epochs,iter_num,patch_size,device,clamp_val)

    print_numeric_results(model,preprocessed_images,label_list,patch,target,device)   

    unormalize = UnNormalize(mean=preprocess.mean,std=preprocess.std)
    unorm_patch = unormalize(patch[0])
    transform = transforms.ToPILImage()
    patch_img = transform(unorm_patch)
    patch_np = unorm_patch.cpu().detach().numpy()

    patch_img.save(os.path.join(output_path,str(target))+".png") 
    np.save(os.path.join(output_path,str(target))+".npy",patch_np)

if __name__ == '__main__':
    main()