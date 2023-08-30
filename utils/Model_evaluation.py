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

def path_2_tensor(path, size:tuple, whether_color=True):
    transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize(mean=0.5, std=0.5)])
    src = cv2.imread(path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) if whether_color else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    tensor = transformer(src)
    return tensor.unsqueeze(0)

def run_model_save(g_f_model, g_c_model, var_list, c_save, f_save, index):
    var_list = map(convert_to_cuda, var_list)
    X_realA, X_realB, X_realA_half, X_realB_half = var_list
    X_fakeB_half, X_global = g_c_model(X_realA_half)
    X_fakeB = g_f_model(X_realA, X_global)
    B_list = [X_realB, X_fakeB, X_realB_half, X_fakeB_half]
    B_list = map(one_to_triple, B_list)
    X_realB, X_fakeB, X_realB_half, X_fakeB_half = B_list
    
    display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0).cpu().detach()
    display_list = (display_list + 1) / 2
    save_image(display_list, os.path.join(c_save, f'{index}.png'))
    
    display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0).cpu().detach()
    display_list = (display_list + 1) / 2
    save_image(display_list, os.path.join(f_save, f'{index}.png'))

def main(args):
    BIGGER_SIZE = (1112, 1448)
    SMALLER_SIZE = (1112//2, 1448//2)
    g_model_coarse = nn.DataParallel(coarse_generator()).cuda()
    g_model_fine = nn.DataParallel(fine_generator()).cuda()
    g_model_fine.module.load_state_dict(torch.load(args.fine_weights))
    g_model_coarse.module.load_state_dict(torch.load(args.coarse_weights))
    c_save_path = check_dir(os.path.join(args.updir, 'Coarse_save'))
    f_save_path = check_dir(os.path.join(args.updir, 'Fine_save'))
    slo_path = os.path.join(args.updir, f'{args.index}.jpg')
    ffa_path = os.path.join(args.updir, f'{args.index}-{args.index}.jpg')
    X_realA = path_2_tensor(slo_path, BIGGER_SIZE)
    X_realA_half = path_2_tensor(slo_path, SMALLER_SIZE)
    X_realB = path_2_tensor(ffa_path, BIGGER_SIZE, whether_color=False)
    X_realB_half = path_2_tensor(ffa_path, SMALLER_SIZE, whether_color=False)
    run_model_save(g_model_fine, g_model_coarse, 
                   [X_realA, X_realB, X_realA_half, X_realB_half], c_save_path, f_save_path, args.index)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--fine_weights', type=str, default='weights/exp_early/g_model_fine.pt')
    parser.add_argument('--coarse_weights', type=str, default='weights/exp_early/g_model_coarse.pt')
    parser.add_argument('--updir', type=str)
    
    args = parser.parse_args()
    main(args)
    
    
    
    