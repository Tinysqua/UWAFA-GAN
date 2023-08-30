import random
from torchvision import transforms
import cv2 as cv
import inspect
import argparse

def check_dir(dire):
    import os
    os.makedirs(dire, exist_ok=True)
    return dire

def get_parameters(fn, original_dict):
    new_dict = dict()
    arg_names = inspect.getfullargspec(fn)[0]
    for k in original_dict.keys():
        if k in arg_names:
            new_dict[k] = original_dict[k]
    return new_dict

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
        
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    
def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)

class Over_var_generator:
    def __init__(self, img_size, up_dir, whole_image_list):
        self.whole_image_list = whole_image_list
        self.up_dir = up_dir
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])
        
        self.transformer_mini = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0]//2, img_size[1]//2)),
            transforms.Normalize(mean=0.5, std=0.5)]) 

    def gene_var_list(self):
        random_int = random.randint(1, len(self.whole_image_list)/2)
        slo_path = f'{self.up_dir}/{random_int}.png'
        ffa_path = f'{self.up_dir}/{random_int}-{random_int}.png'
        slo_pic = cv.imread(slo_path)
        slo_pic = cv.cvtColor(slo_pic, cv.COLOR_BGR2RGB)
        ffa_pic = cv.cvtColor(cv.imread(ffa_path), cv.COLOR_BGR2GRAY)
        slo_pic_whole, ffa_pic_whole = self.transformer(slo_pic).unsqueeze(0), self.transformer(ffa_pic).unsqueeze(0)
        slo_pic_half, ffa_pic_half = self.transformer_mini(slo_pic).unsqueeze(0), self.transformer_mini(ffa_pic).unsqueeze(0)
        return [slo_pic_whole, ffa_pic_whole, slo_pic_half, ffa_pic_half]
    
    def fake_iterator(self):
        iterate_times = 10000 # a huge number to promise the iteration process
        for i in range(iterate_times):
            variable_list = self.gene_var_list()
            yield variable_list
        