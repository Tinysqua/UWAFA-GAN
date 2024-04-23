from PIL import Image  # PIL = Python Image Library
import numpy as np
import os  # 在python下写程序，需要对文件以及文件夹或者其他的进行一系列的操作，os便是对文件或文件夹操作的一个工具。
import random
import argparse  # argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
import glob
from os.path import join as j
import os

def random_crop(img, mask, width, height, num_of_crops,name,stride=1,dir_name='data'):
    Image_dir = j(dir_name, 'Images')  # data/Images
    Mask_dir = j(dir_name, 'Masks')   # data/Masks
    directories = [dir_name,Image_dir,Mask_dir]  # 把它们放入一个数组中
    # 检查文件要写入的目录是否存在？
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    max_y = int(((img.shape[0]-height)/stride)+1)
    max_x = int(((img.shape[1]-width)/stride)+1)

    crop_x = [i for i in range(0,max_x)]
    # 输出0-max_x-1
    crop_y = [i for i in range(0, max_y)]
    # 输出0-max_y-1
    for i in range(1,num_of_crops+1):
        x_seq = random.choice(crop_x)
        y_seq = random.choice(crop_y)

        crop_img_arr = img[y_seq:y_seq+height, x_seq:x_seq+width]

        crop_mask_arr = mask[y_seq:y_seq+height, x_seq:x_seq+width]
        crop_img = Image.fromarray(crop_img_arr)
        crop_mask = Image.fromarray(crop_mask_arr)
        img_name = j(directories[1], f'{name}_{i}.png')
        mask_name = j(directories[2], f'{name}_mask_{i}.png')
        crop_img.save(img_name)
        crop_mask.save(mask_name)


def check_dir(updir, index_interval:list, suffix='.jpg'):
    for i in range(*index_interval):
        slo_address = j(updir, f'{i+1}{suffix}')
        ffa_address = j(updir, f'{i+1}-{i+1}{suffix}')
        af_address = j(updir, f'{i+1}-{i+1}-{i+1}{suffix}')
        address_list = [slo_address, ffa_address]
        
        for k in address_list:
            if not os.path.exists(k):
                print(f'The address {k} doesn\'t exist!')
                return False
    return True
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=1451)
    parser.add_argument('--height', type=int, default=1109)
    parser.add_argument('--datadir', type=str, help='path/to/data_directory', default='dataset/SLO_early_FFA2')
    
    parser.add_argument('--input_dim_width', type=int, default=1088)
    parser.add_argument('--input_dim_height', type=int, default=832)
    parser.add_argument('--n_crops', type=int, default=40)
    parser.add_argument('--output_dir', type=str, default='dataset/data/')
    parser.add_argument('--index_interval', type=list, default=[0, 26], action='append')
    parser.add_argument('--suffix', type=str, default='.png')
    
    args = parser.parse_args()
    
    updir = args.datadir
    new_up_dir = args.output_dir
    size = (args.width, args.height)
    bias = 0
    
    assert check_dir(args.datadir, args.index_interval, args.suffix), 'The index of the pics isn\'t continue'
    
    file_name_list = glob.glob(j(args.datadir, f'*{args.suffix}'))
    len_file = len(file_name_list)
    
    args.index_interval = [i+1+bias for i in args.index_interval]
    for i in range(*args.index_interval):
        old_slo_path = j(updir, str(i) + args.suffix)
        old_ffa_path = j(updir, str(i) + '-' + str(i) + args.suffix)
        old_slo = Image.open(old_slo_path)
        old_ffa = Image.open(old_ffa_path)
        new_slo = np.asarray(old_slo.resize(size))
        new_ffa = np.asarray(old_ffa.resize(size))
        name = str(i)
        random_crop(new_slo, new_ffa, args.input_dim_width, args.input_dim_height, args.n_crops, name, dir_name=args.output_dir)
    
    
    
    