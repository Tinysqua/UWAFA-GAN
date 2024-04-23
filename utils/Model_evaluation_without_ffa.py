import argparse
import sys; sys.path.append('./')
from models.modules import Whole_generator
from utils.common import check_dir
import os
from torchvision.utils import save_image
from torchvision import transforms
import cv2

def path_2_tensor(path, size:tuple, whether_color=True):
    transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize(mean=0.5, std=0.5)])
    src = cv2.imread(path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) if whether_color else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    tensor = transformer(src)
    return tensor.unsqueeze(0)

def main(args):
    BIGGER_SIZE = (1112, 1448)
    SMALLER_SIZE = (1112//2, 1448//2)
    device = args.device
    gen = Whole_generator().to(device)
    gen.load_checkpoints(args.model_updir)
    f_save_path = check_dir(os.path.join(args.updir, f'Fine_save_{args.model_updir.split("/")[-1]}'))
    for i in range(100):
        slo_path = os.path.join(args.updir, f'{i+1}.jpg')
        if not os.path.exists(slo_path):
            break
        X_realA = path_2_tensor(slo_path, BIGGER_SIZE)
        X_realA_half = path_2_tensor(slo_path, SMALLER_SIZE)
        X_realA = X_realA.to(device)
        X_realA_half = X_realA_half.to(device)
        X_fakeB_half, X_fakeB = gen.nograd_run(X_realA_half, X_realA)
        save_image((X_realA+1)/2, os.path.join(f_save_path, f'{i+1}.png'))
        save_image((X_fakeB+1)/2, os.path.join(f_save_path, f'{i+1}-{i+1}.png'))
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--updir', type=str)
    parser.add_argument('--model_updir', type=str, default='weights/exp_8_17')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)