from models.models import coarse_generator, fine_generator
from models.reg import Reg, Transformer_2D
from torch import nn
import torch
from utils.common import check_dir
from os.path import join as j
from models.discriminator_models import MultiscaleDiscriminator
from torch import optim


class Whole_generator(nn.Module):
    def __init__(self, norm_name='instance'):
        super(Whole_generator, self).__init__()
        self.coarser_gen = coarse_generator(norm_type=norm_name)
        self.fine_gen = fine_generator(norm_type=norm_name)
        
    def nograd_run(self, X_realA_half, X_realA):
        with torch.no_grad():
            X_fakeB_half, x_global = self.coarser_gen(X_realA_half)
            X_fakeB = self.fine_gen(X_realA, x_global)
        return X_fakeB_half, X_fakeB
    
    def run(self, X_realA_half, X_realA):
        X_fakeB_half, x_global = self.coarser_gen(X_realA_half)
        X_fakeB = self.fine_gen(X_realA, x_global)
        return X_fakeB_half, X_fakeB
    
    def forward(self, X_realA_half, X_realA, is_no_grad=False):
        if is_no_grad:
            with torch.no_grad():
                X_fakeB_half, x_global = self.coarser_gen(X_realA_half)
                X_fakeB = self.fine_gen(X_realA, x_global)
        else:
            X_fakeB_half, x_global = self.coarser_gen(X_realA_half)
            X_fakeB = self.fine_gen(X_realA, x_global)
        return X_fakeB_half, X_fakeB
            
        
    
    def save_checkpoints(self, updir):
        torch.save(self.state_dict(), j(updir, 'generator.pt'))
        
    def load_checkpoints(self, updir):
        self.load_state_dict(torch.load(j(updir, 'generator.pt')))
        
        
class Whole_discriminator(nn.Module):
    def __init__(self, norm_layer, num_D, num_D_small, n_layers, n_layers_small):
        super(Whole_discriminator, self).__init__()
        self.fine_dis = MultiscaleDiscriminator(input_nc=4, norm_layer=norm_layer, num_D=num_D, 
                                                                n_layers=n_layers, getIntermFeat=True)
        self.coarser_dis = MultiscaleDiscriminator(input_nc=4, ndf=32, norm_layer=norm_layer, num_D=num_D_small, 
                                                                   n_layers=n_layers_small, getIntermFeat=True)
        
    def forward(self, X, fine_level):
        if fine_level:
            d_feat = self.fine_dis(X)
        else:
            d_feat = self.coarser_dis(X)
        return d_feat
        
    def save_checkpoints(self, updir):
        torch.save(self.state_dict(), j(updir, 'discriminator.pt'))
        
    def load_checkpoints(self, updir):
        self.load_state_dict(torch.load(j(updir, 'discriminator.pt')))
        
class Para_combine_trainer:
    def __init__(self, 
                 updir, 
                 norm_name, 
                 num_D, 
                 num_D_small, 
                 n_layers, 
                 n_layers_small, 
                 nlr, 
                 nbeta1, 
                 img_size,
                 use_reg, 
                 gan_loss_computer, 
                 feat_loss_computer, 
                 vgg_loss, 
                 l1_loss, 
                 smooth_loss):
        self.use_reg = use_reg
        self.gen = nn.DataParallel(Whole_generator(norm_name)).cuda()
        norm_layer = nn.InstanceNorm2d if norm_name is 'instance' else nn.BatchNorm2d
        self.dis = nn.DataParallel(Whole_discriminator(norm_layer, num_D, num_D_small, n_layers, n_layers_small)).cuda()
        if self.use_reg:
            self.ra = nn.DataParallel(Reg(img_size[0], img_size[1], 1, 1)).cuda()
            self.spatial_transform = nn.DataParallel(Transformer_2D()).cuda()
        
        self.updir = check_dir(updir)
        self.prepare_training(nlr, nbeta1)
        self.gan_loss_computer = gan_loss_computer
        self.vgg_loss = vgg_loss
        self.feat_loss_computer = feat_loss_computer
        self.l1_loss = l1_loss
        self.smooth_loss = smooth_loss
        self.num_D = num_D
        self.n_layers = n_layers
        self.num_D_small = num_D_small
        self.n_layers_small = n_layers_small
        
    def prepare_training(self, nlr, nbeta1):
        self.optimizerD_f = optim.Adam(self.dis.module.fine_dis.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        self.optimizerD_c = optim.Adam(self.dis.module.coarser_dis.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        self.optimizerG_f = optim.Adam(self.gen.module.fine_gen.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        self.optimizerG_c = optim.Adam(self.gen.module.coarser_gen.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        if self.use_reg:
            self.optim_ra = optim.Adam(self.ra.module.parameters(), lr=nlr//2, betas=(nbeta1, 0.999))
        
    def one_step(self, var_list):
        feat_loss_computer = self.feat_loss_computer
        vgg_loss = self.vgg_loss
        l1_loss = self.l1_loss
        smooth_loss = self.smooth_loss
        X_realA, X_realB, X_realA_half, X_realB_half = map(self.convert_to_cuda, var_list)
        num_D = self.num_D
        n_layers = self.n_layers
        num_D_small = self.num_D_small
        n_layers_small = self.n_layers_small
        dis = self.dis
        gen = self.gen
        if self.use_reg:
            ra = self.ra
            spatial_transform = self.spatial_transform
        
        self.optimizerD_f.zero_grad()
        self.optimizerD_c.zero_grad()
        
        d_feat1_real = dis(torch.cat([X_realA, X_realB], dim=1), True)
        d_loss1 = self.gan_loss_computer(model_output=d_feat1_real, label=True)
        
        X_fakeB_half, X_fakeB = gen(X_realA_half, X_realA, True)
        d_feat1_fake = dis(torch.cat([X_realA, X_fakeB.detach()], dim=1), True)
        d_loss2 = self.gan_loss_computer(model_output=d_feat1_fake, label=False)
        
        d_feat2_real = dis(torch.cat([X_realA_half, X_realB_half], dim=1), False)
        d_loss3 = self.gan_loss_computer(model_output=d_feat2_real, label=True)
        
        d_feat2_fake = dis(torch.cat([X_realA_half, X_fakeB_half.detach()], dim=1), False)
        d_loss4 = self.gan_loss_computer(model_output=d_feat2_fake, label=False)
        
        d_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4
        d_loss.backward()
        
        self.optimizerD_f.step()
        self.optimizerD_c.step()
        
        X_fakeB_half, X_fakeB = gen(X_realA_half, X_realA, False)
        
        if self.use_reg:
            trans = ra(X_fakeB, X_realB)
            sysregist_A2B = spatial_transform(X_fakeB, trans)
        
        d_feat1_real = dis(torch.cat([X_realA, X_realB], dim=1), True)
        d_feat1_fake = dis(torch.cat([X_realA, X_fakeB], dim=1), True)

        d_feat2_real = dis(torch.cat([X_realA_half, X_realB_half], dim=1), False)
        d_feat2_fake = dis(torch.cat([X_realA_half, X_fakeB_half], dim=1), False)
        
        # 2023.8.9 the 'X_fakeB' has been changed to 'sysregist_A2B' to change the vgg loss
        if self.use_reg: 
            variable_list_stacked = X_realB, sysregist_A2B, X_realB_half, X_fakeB_half
        else: 
            variable_list_stacked = X_realB, X_fakeB, X_realB_half, X_fakeB_half
        variable_list_stacked = map(lambda x: torch.cat([x, x, x], dim=1), variable_list_stacked)

        X_realB_stacked, X_fakeB_stacked, X_realB_half_stacked, X_fakeB_half_stacked = variable_list_stacked
        
        self.optimizerG_f.zero_grad()
        self.optimizerG_c.zero_grad()
        
        loss_G_F_GAN = self.gan_loss_computer(model_output=d_feat1_fake, label=True)
        loss_G_F_GAN_Feat = 10*feat_loss_computer((d_feat1_real, d_feat1_fake), num_D=num_D, 
                                                    n_layers=n_layers)
        loss_G_F_VGG = 10 * vgg_loss(X_fakeB_stacked, X_realB_stacked)
        
        loss_G_C_GAN = self.gan_loss_computer(model_output=d_feat2_fake, label=True)  
        loss_G_C_GAN_Feat =  10*feat_loss_computer((d_feat2_real, d_feat2_fake), num_D=num_D_small, 
                                                    n_layers=n_layers_small)
        loss_G_C_VGG = 10 * vgg_loss(X_fakeB_half_stacked, X_realB_half_stacked)
        
        if self.use_reg:
            loss_R_A = 20*l1_loss(sysregist_A2B, X_realB) + 10*smooth_loss(trans)
            gan1_loss = loss_G_F_GAN + loss_G_F_GAN_Feat + loss_G_F_VGG + loss_R_A
            self.optim_ra.zero_grad()
        else:
            gan1_loss = loss_G_F_GAN + loss_G_F_GAN_Feat + loss_G_F_VGG
        gan2_loss = loss_G_C_GAN + loss_G_C_GAN_Feat + loss_G_C_VGG
        

        gan_loss = gan1_loss + gan2_loss
        gan_loss.backward()
        self.optimizerG_f.step()
        self.optimizerG_c.step()
        if self.use_reg:
            self.optim_ra.step()
        
        d_f_loss = d_loss1.item() + d_loss2.item()
        d_c_loss = d_loss3.item() + d_loss4.item()
        gan_loss = gan1_loss.item() + gan2_loss.item()
        
        return d_f_loss, d_c_loss, gan_loss
    
    def save(self):
        self.gen.module.save_checkpoints(self.updir)
        self.dis.module.save_checkpoints(self.updir)
        if self.use_reg:
            self.ra.module.save_checkpoints(self.updir)
        
    def load(self):
        self.gen.module.load_checkpoints(self.updir)
        self.dis.module.load_checkpoints(self.updir)
        if self.use_reg:
            self.ra.module.load_checkpoints(self.updir)
        
    @staticmethod
    def convert_to_cuda(x, device=None):
        if device is None:
            return x.cuda()
        else:
            return x.to(device)
        
        
        
class Combine_trainer:
    def __init__(self, 
                 updir, 
                 norm_name, 
                 num_D, 
                 num_D_small, 
                 n_layers, 
                 n_layers_small, 
                 nlr, 
                 nbeta1, 
                 img_size,
                 use_reg, 
                 gan_loss_computer, 
                 feat_loss_computer, 
                 vgg_loss, 
                 l1_loss, 
                 smooth_loss):
        self.use_reg = use_reg
        self.gen = nn.DataParallel(Whole_generator(norm_name)).cuda()
        norm_layer = nn.InstanceNorm2d if norm_name is 'instance' else nn.BatchNorm2d
        self.dis = nn.DataParallel(Whole_discriminator(norm_layer, num_D, num_D_small, n_layers, n_layers_small)).cuda()
        if self.use_reg:
            self.ra = nn.DataParallel(Reg(img_size[0], img_size[1], 1, 1)).cuda()
            self.spatial_transform = nn.DataParallel(Transformer_2D()).cuda()
        
        self.updir = check_dir(updir)
        self.prepare_training(nlr, nbeta1)
        self.gan_loss_computer = gan_loss_computer
        self.vgg_loss = vgg_loss
        self.feat_loss_computer = feat_loss_computer
        self.l1_loss = l1_loss
        self.smooth_loss = smooth_loss
        self.num_D = num_D
        self.n_layers = n_layers
        self.num_D_small = num_D_small
        self.n_layers_small = n_layers_small
        
    def prepare_training(self, nlr, nbeta1):
        self.optimizerD_f = optim.Adam(self.dis.module.fine_dis.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        self.optimizerD_c = optim.Adam(self.dis.module.coarser_dis.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        self.optimizerG_f = optim.Adam(self.gen.module.fine_gen.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        self.optimizerG_c = optim.Adam(self.gen.module.coarser_gen.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        if self.use_reg:
            self.optim_ra = optim.Adam(self.ra.module.parameters(), lr=nlr//2, betas=(nbeta1, 0.999))
        
    def one_step(self, var_list):
        feat_loss_computer = self.feat_loss_computer
        vgg_loss = self.vgg_loss
        l1_loss = self.l1_loss
        smooth_loss = self.smooth_loss
        X_realA, X_realB, X_realA_half, X_realB_half = map(self.convert_to_cuda, var_list)
        num_D = self.num_D
        n_layers = self.n_layers
        num_D_small = self.num_D_small
        n_layers_small = self.n_layers_small
        dis = self.dis.module
        gen = self.gen.module
        if self.use_reg:
            ra = self.ra.module
            spatial_transform = self.spatial_transform
        
        self.optimizerD_f.zero_grad()
        self.optimizerD_c.zero_grad()
        
        d_feat1_real = dis.fine_dis(torch.cat([X_realA, X_realB], dim=1))
        d_loss1 = self.gan_loss_computer(model_output=d_feat1_real, label=True)
        
        X_fakeB_half, X_fakeB = gen.nograd_run(X_realA_half, X_realA)
        d_feat1_fake = dis.fine_dis(torch.cat([X_realA, X_fakeB.detach()], dim=1))
        d_loss2 = self.gan_loss_computer(model_output=d_feat1_fake, label=False)
        
        d_feat2_real = dis.coarser_dis(torch.cat([X_realA_half, X_realB_half], dim=1))
        d_loss3 = self.gan_loss_computer(model_output=d_feat2_real, label=True)
        
        d_feat2_fake = dis.coarser_dis(torch.cat([X_realA_half, X_fakeB_half.detach()], dim=1))
        d_loss4 = self.gan_loss_computer(model_output=d_feat2_fake, label=False)
        
        d_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4
        d_loss.backward()
        
        self.optimizerD_f.step()
        self.optimizerD_c.step()
        
        X_fakeB_half, X_fakeB = gen.run(X_realA_half, X_realA)
        
        if self.use_reg:
            trans = ra(X_fakeB, X_realB)
            sysregist_A2B = spatial_transform(X_fakeB, trans)
        
        d_feat1_real = dis.fine_dis(torch.cat([X_realA, X_realB], dim=1))
        d_feat1_fake = dis.fine_dis(torch.cat([X_realA, X_fakeB], dim=1))

        d_feat2_real = dis.coarser_dis(torch.cat([X_realA_half, X_realB_half], dim=1))
        d_feat2_fake = dis.coarser_dis(torch.cat([X_realA_half, X_fakeB_half], dim=1))
        
        # 2023.8.9 the 'X_fakeB' has been changed to 'sysregist_A2B' to change the vgg loss
        if self.use_reg: 
            variable_list_stacked = X_realB, sysregist_A2B, X_realB_half, X_fakeB_half
        else: 
            variable_list_stacked = X_realB, X_fakeB, X_realB_half, X_fakeB_half
        variable_list_stacked = map(lambda x: torch.cat([x, x, x], dim=1), variable_list_stacked)

        X_realB_stacked, X_fakeB_stacked, X_realB_half_stacked, X_fakeB_half_stacked = variable_list_stacked
        
        self.optimizerG_f.zero_grad()
        self.optimizerG_c.zero_grad()
        
        loss_G_F_GAN = self.gan_loss_computer(model_output=d_feat1_fake, label=True)
        loss_G_F_GAN_Feat = 10*feat_loss_computer((d_feat1_real, d_feat1_fake), num_D=num_D, 
                                                    n_layers=n_layers)
        loss_G_F_VGG = 10 * vgg_loss(X_fakeB_stacked, X_realB_stacked)
        
        loss_G_C_GAN = self.gan_loss_computer(model_output=d_feat2_fake, label=True)  
        loss_G_C_GAN_Feat =  10*feat_loss_computer((d_feat2_real, d_feat2_fake), num_D=num_D_small, 
                                                    n_layers=n_layers_small)
        loss_G_C_VGG = 10 * vgg_loss(X_fakeB_half_stacked, X_realB_half_stacked)
        
        if self.use_reg:
            loss_R_A = 20*l1_loss(sysregist_A2B, X_realB) + 10*smooth_loss(trans)
            gan1_loss = loss_G_F_GAN + loss_G_F_GAN_Feat + loss_G_F_VGG + loss_R_A
            self.optim_ra.zero_grad()
        else:
            gan1_loss = loss_G_F_GAN + loss_G_F_GAN_Feat + loss_G_F_VGG
        gan2_loss = loss_G_C_GAN + loss_G_C_GAN_Feat + loss_G_C_VGG
        

        gan_loss = gan1_loss + gan2_loss
        gan_loss.backward()
        self.optimizerG_f.step()
        self.optimizerG_c.step()
        if self.use_reg:
            self.optim_ra.step()
        
        d_f_loss = d_loss1.item() + d_loss2.item()
        d_c_loss = d_loss3.item() + d_loss4.item()
        gan_loss = gan1_loss.item() + gan2_loss.item()
        
        return d_f_loss, d_c_loss, gan_loss
    
    def save(self):
        self.gen.module.save_checkpoints(self.updir)
        self.dis.module.save_checkpoints(self.updir)
        if self.use_reg:
            self.ra.module.save_checkpoints(self.updir)
        
    def load(self):
        self.gen.module.load_checkpoints(self.updir)
        self.dis.module.load_checkpoints(self.updir)
        if self.use_reg:
            self.ra.module.load_checkpoints(self.updir)
        
    @staticmethod
    def convert_to_cuda(x, device=None):
        if device is None:
            return x.cuda()
        else:
            return x.to(device)
        
    