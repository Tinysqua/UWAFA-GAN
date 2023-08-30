from torch.utils import data
from torchvision import transforms
import numpy as np
import glob
from os.path import join
import cv2 as cv
try:
    import albumentations as A
except ImportError:
    print('The package albumentations could not be imported')
from torchvision import utils
from PIL import Image
from torch import Generator, randperm
import logging

def get_address_list(up_dir, picture_form: str):
    if up_dir[-1] != '/':
        up_dir = f'{up_dir}/'
    return glob.glob(up_dir+'*.'+picture_form)


class slo_ffa_dataset(data.Dataset):
    def __init__(self, up_dir, img_size, noise_level=0):
        super(slo_ffa_dataset, self).__init__()
        fu_path = join(up_dir, "Images/")
        self.an_path = join(up_dir, "Masks/")
        self.fu_path =  get_address_list(fu_path, "png")
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        # Now we define the transforms in the dataset
        
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.RandomAffine(degrees=2*noise_level, translate=[0.04*noise_level, 0.04*noise_level], 
                                    scale=[1-0.04*noise_level, 1+0.04*noise_level], fill=-1),
            transforms.Normalize(mean=0.5, std=0.5)])
        
        self.transformer_mini = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0]//2, img_size[1]//2)),
            transforms.Normalize(mean=0.5, std=0.5)])    
        
    def __getitem__(self, index):
        fun_filename = self.fu_path[index]
        middle_filename = fun_filename.split("/")[-1].split(".")[0]
        first_num, second_num = int(middle_filename.split("_")[0]), int(middle_filename.split("_")[1])
        
        XReal_A, XReal_A_half = self.convert_to_resize(self.funloader(fun_filename))
        an_filename = str(first_num)+"_mask_"+str(second_num)+".png"
        an_file_path = self.an_path + an_filename
        XReal_B, XReal_B_half = self.convert_to_resize(self.angloader(an_file_path))
        return [XReal_A, XReal_B, XReal_A_half, XReal_B_half]
    
    
    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_mini(X)
        return y1, y2
    
    def __len__(self):
        return len(self.fu_path)
    
    def funloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def angloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
def form_dataloader(data_path, img_size, val_length, seed, batchsize, to_shuffle):
    all_dataset = slo_ffa_dataset(data_path, img_size)
    if val_length > 0:
        data_len = len(all_dataset) - val_length
        logging.info(f'Dataset length: {data_len} and the validation length: {val_length}')
        train_dataset, val_dataset = subset_split(dataset=all_dataset, lengths=[data_len, val_length], generator=Generator().manual_seed(seed))
        train_dataloader = data.DataLoader(train_dataset, batchsize, to_shuffle, drop_last=True)
        val_dataloader = data.DataLoader(val_dataset, 1, True, drop_last=True)
        return train_dataloader, val_dataloader
        
def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(data.Subset(dataset, indices[offset - length : offset]))
    return Subsets
 

class slo_ffa_dataset_auged(data.Dataset):
    def __init__(self, up_dir, img_size):
        super(slo_ffa_dataset_auged, self).__init__()
        fu_path = join(up_dir, "Images/")
        self.an_path = join(up_dir, "Masks/")
        self.fu_path =  get_address_list(fu_path, "png")
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        # Now we define the transforms in the dataset
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])
        
        self.transformer_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0]//2, img_size[1]//2)),
            transforms.Normalize(mean=0.5, std=0.5)])    
        
        self.augmentation = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(), 
            A.ShiftScaleRotate(shift_limit_y=0.02, rotate_limit=40)
        ])
        
    def __getitem__(self, index):
        fun_filename = self.fu_path[index]
        middle_filename = fun_filename.split("/")[-1].split(".")[0]
        first_num, second_num = int(middle_filename.split("_")[0]), int(middle_filename.split("_")[1])
        A_numpy = self.funloader(fun_filename)
        
        an_filename = str(first_num)+"_mask_"+str(second_num)+".png"
        an_file_path = self.an_path + an_filename
        B_numpy = np.expand_dims(self.angloader(an_file_path), axis=2)
        
        AB_cat = np.concatenate((A_numpy, B_numpy), axis=2)
        augmented_AB = self.augmentation(image=AB_cat)['image']
        XReal_A, XReal_A_half = self.convert_to_resize(augmented_AB[:,:,:3])
        XReal_B, XReal_B_half = self.convert_to_resize(augmented_AB[:,:,3])
        return [XReal_A, XReal_B, XReal_A_half, XReal_B_half]
    
    
    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_resize(X)
        return y1, y2
    
    def __len__(self):
        return len(self.fu_path)
    
    def funloader(self, path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
        
    def angloader(self, path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img
        
        
if __name__ == "__main__":
    test_dataset = slo_ffa_dataset('/home/fzj/advanced_VT/dataset/data', (832, 1088))
    t_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    variable_list = next(iter(t_dataloader))
    for i in range(4):
        pic = variable_list[i]
        pic = (pic+1)/2
        utils.save_image(pic, f'aligned/{i}.png')
    


    
