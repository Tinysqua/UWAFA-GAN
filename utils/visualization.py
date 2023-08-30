import random
from os.path import join
from PIL import Image
from torchvision import transforms
import torch
try:
    from visdom import Visdom
except ImportError:
    print('The package visdom could not be imported')
from torch.utils.tensorboard import SummaryWriter



def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)
    
    
def one_to_triple(X, dimension):
    return torch.cat([X, X, X], dim=dimension)




# viz_image = Visdom()
# def iter_summarize_performance(g_f_model, g_c_model, iter_thing, iteration_str):
#     X_realA, X_realB, X_realA_half, X_realB_half = next(iter_thing)
    
    
#     X_realA = convert_to_cuda(X_realA)
#     X_realB = convert_to_cuda(X_realB)
#     X_realA_half = convert_to_cuda(X_realA_half)
#     X_realB_half = convert_to_cuda(X_realB_half)
    
#     X_fakeB_half, X_global = g_c_model(X_realA_half)
#     X_fakeB = g_f_model(X_realA, X_global)
    
#     X_realB = one_to_triple(X_realB, dimension=1)
#     X_fakeB = one_to_triple(X_fakeB, dimension=1)
#     X_realB_half = one_to_triple(X_realB_half, dimension=1)
#     X_fakeB_half = one_to_triple(X_fakeB_half, dimension=1)
    
#     display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0).cpu().detach()
#     display_list = (display_list + 1) / 2
    
#     viz_image.images(display_list, env="VT_global", opts=dict(title= iteration_str), nrow=1)
    
#     display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0).cpu().detach()
#     display_list = (display_list + 1) / 2
    
#     viz_image.images(display_list, env="VT_local", opts=dict(title= iteration_str), nrow=1)

class Visualizer:
    def __init__(self, weights_up_dir, way='tensorboard'):
        if way == 'tensorboard':
            self.recorder = SummaryWriter(weights_up_dir)
            self.use_tensorboard = True
        elif way == 'visdom':
            self.recorder = Visdom()
            self.use_tensorboard = False
        else:
            raise NotImplementedError('Visulizer [%s] is not implemented' % way)
        
    def scalars_initialize(self):
        if not self.use_tensorboard:
            self.recorder.line([[5.], [5.], [70.]], [0], win="VTGAN_LOSS", opts=dict(title='loss',
                                                                       legend=['d_f_loss', 'd_c_loss', 'gan_loss']))
            self.recorder.line([[70.]], [0], win="Fid_score", opts=dict(title='Fid',
                                                                       legend=['fid']))
            self.recorder.line([[0.], [0.]], [0], win="Kid_score", opts=dict(title='Kid',
                                                                   legend=['kid_mean', 'kid_std']))
            
    def scalar_adjuster(self, values, step, title, legend=None):
        if self.use_tensorboard:
            self.tb_draw_scalars(values, step, title, legend)
        else:
            self.viz_draw_scalars(values, step, title)
            
    def tb_draw_scalars(self, values, step, title, legend):
        self.recorder.add_scalars(main_tag=title, 
                                  tag_scalar_dict=dict(zip(legend, values)), global_step=step)
        self.recorder.flush()
        
            
    def viz_draw_scalars(self, values, step, title):
        value_len = len(values)
        visdom_list = []
        for i in range(value_len):
            visdom_list.append([values[i]])
        self.recorder.line(visdom_list, step, win=title, update='append')
        
        
    # draw images per epoch    
    def iter_summarize_performance(self, g_f_model, g_c_model, iter_thing, iteration_str, combine_trainer=None):
        X_realA, X_realB, X_realA_half, X_realB_half = next(iter_thing)
        env_tag = ("VT_global", "VT_local")
        
        X_realA = convert_to_cuda(X_realA)
        X_realB = convert_to_cuda(X_realB)
        X_realA_half = convert_to_cuda(X_realA_half)
        X_realB_half = convert_to_cuda(X_realB_half)
        
        if combine_trainer is None:
            X_fakeB_half, X_global = g_c_model(X_realA_half)
            X_fakeB = g_f_model(X_realA, X_global)
        else:
            X_fakeB_half, X_fakeB = combine_trainer.gen.module.nograd_run(X_realA_half, X_realA)
        
        
        X_realB = one_to_triple(X_realB, dimension=1)
        X_fakeB = one_to_triple(X_fakeB, dimension=1)
        X_realB_half = one_to_triple(X_realB_half, dimension=1)
        X_fakeB_half = one_to_triple(X_fakeB_half, dimension=1)
        
        display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0).cpu().detach()
        display_list = (display_list + 1) / 2
        
        if self.use_tensorboard:
            self.recorder.add_images(env_tag[0], display_list, iteration_str) 
            self.recorder.flush()
        else:
            self.recorder.images(display_list, env=env_tag[0], opts=dict(title= iteration_str), nrow=1)
        
        display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0).cpu().detach()
        display_list = (display_list + 1) / 2
        
        if self.use_tensorboard:
            self.recorder.add_images(env_tag[1], display_list, iteration_str)
            self.recorder.flush() 
        else:
            self.recorder.images(display_list, env=env_tag[1], opts=dict(title= iteration_str), nrow=1)
            
    def close_recorder(self):
        if self.use_tensorboard:
            self.recorder.close()
 