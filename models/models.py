import torch
from torch import nn


class encoder_block(torch.nn.Module):
    def __init__(self, input_channel, down_filter, norm_type='batch'):
        super(encoder_block, self).__init__()
        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm_layer [%s] is not implemented' % norm_type)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.Conv_3_2 = torch.nn.Conv2d(in_channels=input_channel, out_channels=down_filter, kernel_size=(3, 3), stride=2, padding=1)
        self.norm_layer = self.norm(down_filter)

    def forward(self, X):
        X = self.Conv_3_2(X)
        X = self.norm_layer(X)
        X = self.LeakyReLU(X)
        return X
    
class decoder_block(torch.nn.Module):
    def __init__(self, input_channel, up_filter, norm_type='batch'):
        super(decoder_block, self).__init__()
        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm_layer [%s] is not implemented' % norm_type)
        self.convT = torch.nn.ConvTranspose2d(in_channels=input_channel, out_channels=up_filter, kernel_size=(3, 3), stride=2, padding=1,output_padding=1)
        self.norm_layer = self.norm(up_filter)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
    def forward(self,X):
        X = self.convT(X)
        X = self.norm_layer(X)
        X = self.LeakyReLU(X)
        return X

class SeparableConv2D(torch.nn.Module):
    def __init__(self, filters_in=256, filters_out=256, dilation_r=1, padding=0):
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
    def __init__(self, filters, Separable=True, norm_type='batch'):
        super(novel_residual_block, self).__init__()
        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm_layer [%s] is not implemented' % norm_type)
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
            
        self.norm_layer_1 = self.norm(filters)
        self.norm_layer_2 = self.norm(filters)
        self.norm_layer_3 = self.norm(filters)
        # self.BatchNorm2d_2 = torch.nn.BatchNorm2d(filters)
        # self.BatchNorm2d_3 = torch.nn.BatchNorm2d(filters)

    def forward(self, X_input):
        X = X_input
        X = self.Re_Pad_or_id(X)
        X = self.S_or_nor_Conv2D_1(X)
        X = self.norm_layer_1(X)
        X = self.LeakyReLU(X)

        X_branch_1 = self.Re_Pad_or_id(X)
        X_branch_1 = self.S_or_nor_Conv2D_2(X_branch_1)
        X_branch_1 = self.norm_layer_2(X_branch_1)
        X_branch_1 = self.LeakyReLU(X_branch_1)

        ## Branch 2
        X_branch_2 = self.Re_Pad_or_id2(X)
        X_branch_2 = self.S_or_nor_Conv2D_3(X_branch_2)
        X_branch_2 = self.norm_layer_3(X_branch_2)
        X_branch_2 = self.LeakyReLU(X_branch_2)
        X_add_branch_1_2 = torch.add(X_branch_2, X_branch_1)
        X = torch.add(X_input, X_add_branch_1_2)
        return X




class Attention(torch.nn.Module):
    def __init__(self, input_channels, filters, norm_type='batch'):
        super(Attention, self).__init__()
        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm_layer [%s] is not implemented' % norm_type)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.norm_layer_1 = self.norm(filters)
        self.norm_layer_2 = self.norm(filters)
        self.Conv_3_1_first = torch.nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(3, 3), padding=1)
        self.Conv_3_1_second = torch.nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=(3, 3), padding=1)
    
    def forward(self, X):
        X_input = X
        X = self.Conv_3_1_first(X)
        X = self.norm_layer_1(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        
        X = self.Conv_3_1_second(X)
        X = self.norm_layer_2(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        return X
    
# fine -> local -> full/not half
class fine_generator(torch.nn.Module):
    def __init__(self, nff=64, n_blocks=3, use_separable=False, norm_type='instance'):
        super(fine_generator, self).__init__()
        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm_layer [%s] is not implemented' % norm_type)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        # self.n_coarse_gen = n_coarse_gen
        self.Conv_7_1 = torch.nn.Conv2d(in_channels=3, out_channels=nff, kernel_size=(7, 7), padding=0)
        self.Conv_7_1_2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7, 7), padding=0)

        self.ReflectionPad3 = torch.nn.ReflectionPad2d(3)

        self.norm_layer_64 = self.norm(64)
        self.norm_layer_128 = self.norm(128)

        self.encoder_block1 = encoder_block(64, 128, norm_type=norm_type)
        self.encoder_block2 = encoder_block(128, 256, norm_type=norm_type)
        self.encoder_block3 = encoder_block(256, 512, norm_type=norm_type)
        self.middle_Conv2D_1 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3, padding=0), self.norm(512)])
        self.decoder_block1 = decoder_block(512, 256, norm_type=norm_type)
        self.decoder_block2 = decoder_block(512, 128, norm_type=norm_type)
        self.decoder_block3 = decoder_block(256, 64, norm_type=norm_type)
        
        self.Attention1 = Attention(64, 64, norm_type=norm_type)
        self.Attention2 = Attention(128, 128, norm_type=norm_type)
        self.Attention3 = Attention(256, 256, norm_type=norm_type)
        
        residual_list = []
        self.n_blocks = n_blocks
        for _ in range(n_blocks-1):
            residual_list.append(novel_residual_block(512, Separable=use_separable, norm_type=norm_type))
        self.Residual_block = torch.nn.Sequential(*residual_list)

    def forward(self, X_input:torch.Tensor, X_coarse:list):
        # Downsampling layers
        X = self.ReflectionPad3(X_input)
        X = self.Conv_7_1(X)
        X = self.norm_layer_64(X)
        X = self.LeakyReLU(X)

        X_down1 = self.encoder_block1(X)
        
        
        X_down2 = self.encoder_block2(X_down1)
        X= torch.add(X_coarse[0], X_down2)
        
        X_down3 = self.encoder_block3(X)
        X= torch.add(X_coarse[1], X_down3)
        
        X = self.middle_Conv2D_1(X_down3)
        X = self.LeakyReLU(X)

        
        X = self.Residual_block(X)
        
        X += X_coarse[2]
        X_up1 = self.decoder_block1(X)
        X_up1 += X_coarse[3]
        X_up1_att = self.Attention3(X_down2)
        X_up1_add = torch.cat([X_up1_att, X_up1], dim=1)
        
        X_up2 = self.decoder_block2(X_up1_add)
        X_up2_att = self.Attention2(X_down1)
        X_up2_add = torch.cat([X_up2_att, X_up2], dim=1)
        
        X_up3 = self.decoder_block3(X_up2_add)
        
        X = self.ReflectionPad3(X_up3)
        X = self.Conv_7_1_2(X)
        X = torch.tanh(X)
        return X
    

# coarse -> global -> half
class coarse_generator(torch.nn.Module):
    def __init__(self, ncf=64, n_downsampling=2, n_blocks=9, use_separable=False, norm_type='instance'):
        super(coarse_generator, self).__init__()
        if norm_type == 'batch':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm2d
        else:
            raise NotImplementedError('norm_layer [%s] is not implemented' % norm_type)
        self.ncf=ncf
        self.n_blocks = n_blocks
        self.n_downsampling=n_downsampling

        self.Conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=ncf, kernel_size=(7, 7), padding=0)
        self.Conv_2 = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(7, 7), padding=0)

        self.norm_layer_64 = self.norm(64)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

        self.ReflectionPad3 = torch.nn.ReflectionPad2d(3)

        self.decoder_block1 = decoder_block(512, 256, norm_type=norm_type)

        self.decoder_block2 = decoder_block(512, 64, norm_type=norm_type)

        self.Attention1 = Attention(256, 256, norm_type=norm_type)
        self.Attention2 = Attention(64, 64, norm_type=norm_type)

        self.encoder_block1 = encoder_block(64, 256, norm_type=norm_type)

        self.encoder_block2 = encoder_block(256, 512, norm_type=norm_type)
        
        residual_list = []
        for _ in range(n_blocks):
            residual_list.append(novel_residual_block(filters=512, Separable=use_separable, norm_type=norm_type))
        self.novel_Residual_block1 = torch.nn.Sequential(*residual_list)

    def forward(self,X_input):
        feature_out = []
        X = self.ReflectionPad3(X_input)
        X = self.Conv_1(X)
        X = self.norm_layer_64(X)
        X_pre_down = self.LeakyReLU(X)

        # Downsampling layers
        X_down1 = self.encoder_block1(X)
        feature_out.append(X_down1)
        X_down2 = self.encoder_block2(X_down1)
        feature_out.append(X_down2)
        X = X_down2

        
        X = self.novel_Residual_block1(X)
        feature_out.append(X)
        # Upsampling layers
        X_up1 = self.decoder_block1(X)
        X_up1_att = self.Attention1(X_down1)
        X_up1_add = torch.cat([X_up1_att, X_up1], dim=1)
        feature_out.append(X_up1)
        
        X_up2 = self.decoder_block2(X_up1_add)
        X_up2_att = self.Attention2(X_pre_down)
        X_up2_add = torch.cat([X_up2_att, X_up2], dim=1)

        X = self.ReflectionPad3(X_up2_add)
        X =self.Conv_2(X)
        X = torch.tanh(X)
        return X, feature_out


if __name__ == "__main__":
    a = torch.randn((1, 3, 416, 544))
    b = torch.randn((1, 3, 416*2, 544*2))
    model = coarse_generator()
    model_model = fine_generator()
    _, feature = model(a)
    model_model(b, feature)
    for i in feature:
        print("Shape: ", i.shape)
    
