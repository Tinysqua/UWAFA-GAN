import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import sys
sys.path.append("../advanced_VT/")
from models.models import *
from torch import nn
import os
from models.modules import Whole_generator
from models.reg import Reg, Transformer_2D
from functools import partial

def check_dir(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
    return dire

def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)

def one_to_triple(X):
    return torch.cat([X, X, X], dim=1)

def get_tensor(slo_path, ffa_path, BIGGER_SIZE, SMALLER_SIZE, affine, noise_level):
    if not affine:
        X_realA = path_2_tensor(slo_path, BIGGER_SIZE)
        X_realA_half = path_2_tensor(slo_path, SMALLER_SIZE)
        X_realB = path_2_tensor(ffa_path, BIGGER_SIZE, whether_color=False)
        X_realB_half = path_2_tensor(ffa_path, SMALLER_SIZE, whether_color=False)
    else:
        X_realA, X_realB = affine_path_2_tensor([slo_path, ffa_path], BIGGER_SIZE, noise_level)
        X_realA_half, X_realB_half = affine_path_2_tensor([slo_path, ffa_path], SMALLER_SIZE, noise_level)
    return [X_realA, X_realB, X_realA_half, X_realB_half]

def path_2_tensor(path, size:tuple, whether_color=True):
    transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize(mean=0.5, std=0.5)])
    src = cv2.imread(path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) if whether_color else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    tensor = transformer(src)
    return tensor.unsqueeze(0)

def affine_path_2_tensor(two_path:list, size:tuple, noise_level, affine_type='RandomAffine'):
    '''
    two_path: First is slo, second ffa
    size: what size I want to resize the tensor to 
    noise_level: the noise level of the affine activation
    '''
    transformer_list = [transforms.Resize(size), transforms.Normalize(mean=0.5, std=0.5)]
    if affine_type == 'RandomAffine':
        transformer_list.append(transforms.RandomAffine(degrees=5*noise_level, translate=[
                                     0.04*noise_level, 0.04*noise_level], scale=[1-0.04*noise_level, 1+0.04*noise_level], fill=-1))
    to_tensor = transforms.ToTensor()
    transformer = transforms.Compose(transformer_list)
    slo = cv2.cvtColor(cv2.imread(two_path[0]), cv2.COLOR_BGR2RGB)
    ffa = cv2.cvtColor(cv2.imread(two_path[1]), cv2.COLOR_BGR2GRAY)
    tensor = torch.cat([to_tensor(slo), to_tensor(ffa)], dim=0)
    tensor = transformer(tensor).unsqueeze(0)
    return tensor[:,0:3,...], tensor[:,3:,...]
    

def run_model_save(gen, var_list, c_save, f_save, index):
    var_list = map(convert_to_cuda, var_list)
    X_realA, X_realB, X_realA_half, X_realB_half = var_list
    X_fakeB_half, X_fakeB = gen.nograd_run(X_realA_half, X_realA)
    B_list = [X_realB, X_fakeB, X_realB_half, X_fakeB_half]
    B_list = map(one_to_triple, B_list)
    X_realB, X_fakeB, X_realB_half, X_fakeB_half = B_list
    
    display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0).cpu().detach()
    display_list = (display_list + 1) / 2
    save_image(display_list, os.path.join(c_save, f'{index}.png'))
    
    display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0).cpu().detach()
    display_list = (display_list + 1) / 2
    save_image(display_list, os.path.join(f_save, f'{index}.png'))
    
def affine_model_save(gen, ra, trans2d, var_list, c_save, f_save, index, num=1):
    X_realA, X_realB, X_realA_half, X_realB_half = map(convert_to_cuda, var_list)
    X_fakeB_half, X_fakeB = gen.nograd_run(X_realA_half, X_realA)
    with torch.no_grad():
        trans = ra(X_fakeB, X_realB)
        sysregist_A2B = trans2d(X_fakeB, trans)
    B_list = [X_realB, X_fakeB, X_realB_half, X_fakeB_half, sysregist_A2B]
    B_list = map(one_to_triple, B_list)
    X_realB, X_fakeB, X_realB_half, X_fakeB_half, sysregist_A2B = B_list
    
    display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0).cpu().detach()
    display_list = (display_list + 1) / 2
    save_image(display_list, os.path.join(c_save, f'{index}_{num}.png'))
    
    display_list = torch.cat([X_realA, X_fakeB, X_realB, sysregist_A2B], dim=0).cpu().detach()
    display_list = (display_list + 1) / 2
    save_image(display_list, os.path.join(f_save, f'{index}_{num}.png'))
    
def main(args):
    BIGGER_SIZE = (1112, 1448)
    SMALLER_SIZE = (1112//2, 1448//2)
    device = args.device
    gen = Whole_generator().to(device)
    gen.load_checkpoints(args.model_updir)
    if args.affine:
        ra = Reg(BIGGER_SIZE[0], BIGGER_SIZE[1], 1, 1).to(device)
        ra.load_checkpoints(args.model_updir)
    c_save_path = check_dir(os.path.join(args.updir, 'Coarse_save'))
    f_save_path = check_dir(os.path.join(args.updir, 'Fine_save'))
    slo_path = os.path.join(args.updir, f'{args.index}.jpg')
    ffa_path = os.path.join(args.updir, f'{args.index}-{args.index}.jpg')
    if not args.affine:
        var_list = get_tensor(slo_path, ffa_path, BIGGER_SIZE, SMALLER_SIZE, args.affine, args.noise_level)
        run_model_save(gen, var_list, 
                    c_save_path, f_save_path, args.index)
    else:
        trans2d = Transformer_2D().to(device)
        for i in range(args.affine):
            var_list = get_tensor(slo_path, ffa_path, BIGGER_SIZE, SMALLER_SIZE, args.affine, args.noise_level)
            affine_model_save(gen, ra, trans2d, var_list, c_save_path, f_save_path, args.index, num=i+1)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--updir', type=str)
    parser.add_argument('--model_updir', type=str, default='weights/exp_8_17/')
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--noise_level', default=0)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    main(args)
    
    
    
    