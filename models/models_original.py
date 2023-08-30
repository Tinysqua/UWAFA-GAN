import torch
from torch import nn


class encoder_block(torch.nn.Module):
    def __init__(self,input_channel,down_filter):
        super(encoder_block, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.Conv_3_2 = torch.nn.Conv2d(in_channels=input_channel, out_channels=down_filter, kernel_size=(3, 3), stride=2, padding=1)
        self.BatchNorm2d = torch.nn.BatchNorm2d(down_filter)
    def forward(self,X):
        X = self.Conv_3_2(X)
        X = self.BatchNorm2d(X)
        X = self.LeakyReLU(X)
        return X
    
class decoder_block(torch.nn.Module):
    def __init__(self,input_filter,filter):
        super(decoder_block, self).__init__()
        self.convT = torch.nn.ConvTranspose2d(in_channels=input_filter, out_channels=filter, kernel_size=(3, 3), stride=2, padding=1,output_padding=1)
        self.BatchNorm2d = torch.nn.BatchNorm2d(filter)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
    def forward(self,X):
        X = self.convT(X)
        X = self.BatchNorm2d(X)
        X = self.LeakyReLU(X)
        return X

class SeparableConv2D(torch.nn.Module):
    def __init__(self,filters_in=256,filters_out=256,dilation_r=1,padding=0):
        super(SeparableConv2D, self).__init__()
        self.depth_conv = torch.nn.Conv2d(in_channels=filters_in, out_channels=filters_in,
                                    kernel_size=(3,3),groups=filters_in,padding=padding,dilation=dilation_r)
        self.point_conv = torch.nn.Conv2d(in_channels=filters_in, out_channels=filters_out,
                                    kernel_size=(1,1),dilation=1)

    def forward(self,X):

        out = self.depth_conv(X)
        out = self.point_conv(out)
        return out
    

class novel_residual_block(torch.nn.Module):
    def __init__(self,filters,Separable=True):
        super(novel_residual_block, self).__init__()
        
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

        if Separable:
            self.Re_Pad_or_id = torch.nn.ReflectionPad2d(1)
            self.Re_Pad_or_id2 = torch.nn.ReflectionPad2d(2)
            self.S_or_nor_Conv2D_1 = SeparableConv2D(filters_in=filters, filters_out=filters, dilation_r=1)
            self.S_or_nor_Conv2D_2 = SeparableConv2D(filters_in=filters, filters_out=filters, dilation_r=1)
            self.S_or_nor_Conv2D_3 = SeparableConv2D(filters_in=filters, filters_out=filters, dilation_r=2)
        else:
            self.Re_Pad_or_id = nn.Identity()
            self.Re_Pad_or_id2 = nn.Identity()
            self.S_or_nor_Conv2D_1 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(filters, filters, kernel_size=3, padding=0)])
            self.S_or_nor_Conv2D_2 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(filters, filters, kernel_size=3, padding=0)])
            self.S_or_nor_Conv2D_3 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(filters, filters, kernel_size=3, padding=0)])
            
        self.BatchNorm2d_1 = torch.nn.BatchNorm2d(filters)
        self.BatchNorm2d_2 = torch.nn.BatchNorm2d(filters)
        self.BatchNorm2d_3 = torch.nn.BatchNorm2d(filters)
    def forward(self,X_input):
        X = X_input
        X = self.Re_Pad_or_id(X)
        X = self.S_or_nor_Conv2D_1(X)
        X = self.BatchNorm2d_1(X)
        X = self.LeakyReLU(X)

        X_branch_1 = self.Re_Pad_or_id(X)
        X_branch_1 = self.S_or_nor_Conv2D_2(X_branch_1)
        X_branch_1 = self.BatchNorm2d_2(X_branch_1)
        X_branch_1 = self.LeakyReLU(X_branch_1)

        ## Branch 2
        X_branch_2 = self.Re_Pad_or_id2(X)
        X_branch_2 = self.S_or_nor_Conv2D_3(X_branch_2)
        X_branch_2 = self.BatchNorm2d_3(X_branch_2)
        X_branch_2 = self.LeakyReLU(X_branch_2)
        X_add_branch_1_2 = torch.add(X_branch_2, X_branch_1)
        X = torch.add(X_input, X_add_branch_1_2)
        return X




class Attention(torch.nn.Module):
    def __init__(self,input_channels,filters):
        super(Attention, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.BatchNorm_1 = torch.nn.BatchNorm2d(filters)
        self.BatchNorm_2 = torch.nn.BatchNorm2d(filters)
        self.Conv_3_1_first = torch.nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(3, 3), padding=1)
        self.Conv_3_1_second = torch.nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(3, 3), padding=1)
    
    def forward(self,X):
        X_input = X
        X = self.Conv_3_1_first(X)
        X = self.BatchNorm_1(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        
        X = self.Conv_3_1_second(X)
        X = self.BatchNorm_2(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        return X
    

class fine_generator(torch.nn.Module):
    def __init__(self, nff=64, n_coarse_gen=1, n_blocks=3, use_separable=False):
        super(fine_generator, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.n_coarse_gen = n_coarse_gen
        self.n_blocks = n_blocks
        self.Conv_7_1 = torch.nn.Conv2d(in_channels=3, out_channels=nff, kernel_size=(7, 7), padding=0)
        self.Conv_7_1_2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7, 7), padding=0)

        self.ReflectionPad3 = torch.nn.ReflectionPad2d(3)

        self.BatchNorm2d_64 = torch.nn.BatchNorm2d(64)
        self.BatchNorm2d_128 = torch.nn.BatchNorm2d(128)

        self.encoder_block1 = encoder_block(64, 64)
        self.middle_Conv2D_1 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3, padding=0)])
        self.novel_residual_block1 = novel_residual_block(128)
        self.decoder_block1 = decoder_block(128, 64)
        self.Attention1 = Attention(64, 64)
        
        residual_list = []
        for _ in range(n_blocks-1):
            residual_list.append(novel_residual_block(128, Separable=use_separable))
        self.Residual_block = torch.nn.Sequential(*residual_list)

    def forward(self,X_input, X_coarse):
        # Downsampling layers
        X = self.ReflectionPad3(X_input)
        X = self.Conv_7_1(X)
        X = self.BatchNorm2d_64(X)
        X_pre_down = self.LeakyReLU(X)

        X_down1 = self.encoder_block1(X)
        X= torch.add(X_coarse, X_down1)
        
        X = self.middle_Conv2D_1(X)
        X = self.BatchNorm2d_128(X)
        X = self.LeakyReLU(X)

        
        X = self.Residual_block(X)
        
        X_up1 = self.decoder_block1(X)
        X_up1_att = self.Attention1(X_pre_down)
        X_up1_add = torch.add(X_up1_att, X_up1)
        X = self.ReflectionPad3(X_up1_add)
        X = self.Conv_7_1_2(X)
        X = torch.tanh(X)
        return X
    


class coarse_generator(torch.nn.Module):
    def __init__(self, ncf=64, n_downsampling=2, n_blocks=9, use_separable=False):
        super(coarse_generator, self).__init__()
        self.ncf=ncf
        self.n_blocks = n_blocks
        self.n_downsampling=n_downsampling

        self.Conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=ncf, kernel_size=(7, 7), padding=0)
        self.Conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7, 7), padding=0)

        self.BatchNorm2d_64 = torch.nn.BatchNorm2d(64)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

        self.ReflectionPad3 = torch.nn.ReflectionPad2d(3)

        up_filters = int(self.ncf * pow(2, (self.n_downsampling - 0)) / 2)
        self.decoder_block1 = decoder_block(256, up_filters)

        up_filters_2 = int(self.ncf * pow(2, (self.n_downsampling - 1)) / 2)
        self.decoder_block2 = decoder_block(128, up_filters_2)

        self.Attention1 = Attention(128, 128)
        self.Attention2 = Attention(64, 64)

        down_filters_1 = 64 * pow(2, 0) * 2
        self.encoder_block1 = encoder_block(64, down_filters_1)

        down_filters_2 = 64 * pow(2, 1) * 2
        self.encoder_block2 = encoder_block(128, down_filters_2)
        res_filters = pow(2, n_downsampling)
        residual_list = []
        for _ in range(n_blocks):
            residual_list.append(novel_residual_block(filters=ncf*res_filters, Separable=use_separable))
        self.novel_Residual_block1 = torch.nn.Sequential(*residual_list)

    def forward(self,X_input):
        X = self.ReflectionPad3(X_input)
        X = self.Conv_1(X)
        X = self.BatchNorm2d_64(X)
        X_pre_down = self.LeakyReLU(X)

        # Downsampling layers
        X_down1 = self.encoder_block1(X)
        X_down2 = self.encoder_block2(X_down1)
        X = X_down2

        
        X = self.novel_Residual_block1(X)

        # Upsampling layers

        X_up1 = self.decoder_block1(X)
        X_up1_att = self.Attention1(X_down1)
        X_up1_add = torch.add(X_up1_att, X_up1)

        X_up2 = self.decoder_block2(X_up1_add)
        X_up2_att = self.Attention2(X_pre_down)
        X_up2_add = torch.add(X_up2_att, X_up2)
        feature_out = X_up2_add

        X = self.ReflectionPad3(X_up2_add)
        X =self.Conv_2(X)
        X = torch.tanh(X)
        return X, feature_out



