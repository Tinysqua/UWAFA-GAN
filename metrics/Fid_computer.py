import sys
sys.path.append("../advanced_VT/")
from dataloader.Final_dataloader import Evaluation_dataset
from torch.utils import data

import torch
from torchvision import transforms
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance


from models.models import *

class Kid_Or_Fid:
    def __init__(self, if_cuda=True):
        self.g_model_coarse = None
        self.g_model_fine = None
        F_A_dataset = Evaluation_dataset("dataset/", (1112, 1448))
        self.test_loader = data.DataLoader(F_A_dataset, batch_size=2)
        subset_size = len(self.test_loader)
        
        self.resize_tran = transforms.Resize((299, 299)) 
        self.if_cuda = if_cuda
        if if_cuda:
            self.kid_model = KernelInceptionDistance(normalize=True, subset_size=subset_size).cuda()  
            self.fid_model = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
        else:
            self.kid_model = KernelInceptionDistance(normalize=True, subset_size=subset_size)  
            self.fid_model = FrechetInceptionDistance(feature=2048, normalize=True)   
        
        self.X_realB_list = []
        self.X_fakeB_list = []
        
        self.cat_flag = False
        
    def spin_once(self):
        for variable in self.test_loader:
            self.model_forward(variable)
        fid_score = self.compute(compute_way='fid')
        kid_score = self.compute()
        self.reset()
        return fid_score.item(), kid_score[0].item(), kid_score[1].item()
    
    def reset(self):
        self.X_realB_list = []
        self.X_fakeB_list = []
        self.cat_flag = False 
        
    def model_forward(self, variable_lists):
        variable_lists = map(self.convert_to_cuda, variable_lists)
        X_realA, X_realB, X_realA_half, X_realB_half = variable_lists
        with torch.no_grad():
            X_fakeB_half, X_global = self.g_model_coarse(X_realA_half)
            X_fakeB = self.g_model_fine(X_realA, X_global)
            
        B_lists = [X_realB, X_fakeB, X_realB_half, X_fakeB_half]
        B_lists = map(lambda x:torch.cat([x, x, x], dim=1), B_lists)
        B_lists = map(lambda x:(x+1)/2, B_lists)
        X_realB, X_fakeB, X_realB_half, X_fakeB_half = B_lists
        self.X_realB_list.append(X_realB if self.if_cuda else X_realB.cpu())
        self.X_fakeB_list.append(X_fakeB if self.if_cuda else X_fakeB.cpu())
        
    def compute(self, compute_way = 'kid'):
        if not self.cat_flag:
            self.X_realB_list = self.resize_tran(torch.cat(self.X_realB_list))
            self.X_fakeB_list = self.resize_tran(torch.cat(self.X_fakeB_list))
            self.cat_flag = True
        if compute_way == 'kid':
            self.kid_model.update(self.X_realB_list, real=True)
            self.kid_model.update(self.X_fakeB_list, real=False)
            kid_mean, kid_std = self.kid_model.compute()
            self.kid_model.reset()
            return (kid_mean.cpu(), kid_std.cpu()) if self.if_cuda else (kid_mean, kid_std)
        elif compute_way == 'fid':
            self.fid_model.update(self.X_realB_list, real=True)
            self.fid_model.update(self.X_fakeB_list, real=False)
            fid_value = self.fid_model.compute()
            self.fid_model.reset()
            return fid_value.cpu() if self.if_cuda else fid_value
        else:
            raise NotImplementedError('Couldn\'t find a compute way')
    
        
    def convert_to_cuda(self, x, device=None):
        if device==None:
            return x.cuda()
        else:
            return x.to(device)
    
    def update_models(self, g_fine_model, g_coarse_model):
        self.g_model_fine = g_fine_model
        self.g_model_coarse = g_coarse_model
  