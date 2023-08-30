import argparse
import yaml
from dataloader import VT_dataloader
from models.models import *
from models.modules import Combine_trainer, Para_combine_trainer
from utils.VTGAN_loss import *
from torch import nn
from utils.visualization import Visualizer
from shutil import copyfile
import logging
from metrics.Fid_computer import Kid_Or_Fid
from functools import partial
from utils.common import get_parameters, check_dir
from os.path import join

logging.basicConfig(filename='run.log', datefmt="%Y-%m-%d %H:%M:%S", filemode='w', level=logging.INFO,
                    format="%(asctime)s | %(message)s")

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config
    
def main(args):
    train_config = load_config(args.model_config_path)
    train_config["img_size"] = tuple(train_config["img_size"])
    
    
    vgg_loss = VGGLoss()
    mse = nn.MSELoss()
    l1_loss = nn.L1Loss()
    gan_loss_computer = partial(Discriminator_loss_computer, loss_fn=mse, device_fn=convert_to_cuda)
    feat_loss_computer = partial(Feat_loss_computer, loss_fn=l1_loss)
    train_config['vgg_loss'] = vgg_loss
    train_config['gan_loss_computer'] = gan_loss_computer
    train_config['feat_loss_computer'] = feat_loss_computer
    train_config['num_D_small'] = train_config['num_D'] // 2
    train_config['n_layers_small'] = train_config['n_layers']
    train_config['l1_loss'] = l1_loss
    train_config['smooth_loss'] = smoothing_loss
    
    train_loader, val_loader = VT_dataloader.form_dataloader(**get_parameters(VT_dataloader.form_dataloader, train_config))
    len_along_epoch = len(train_loader)
    
    model = Para_combine_trainer(**get_parameters(Para_combine_trainer, train_config))
    # this step will create the corresponding parrent dir
    save_updir = train_config['updir']
    cfg_base_name = save_updir.split('/')[-1]
    visualizer = Visualizer(weights_up_dir=check_dir(join(save_updir, 'tb_result')), way='tensorboard')
    copyfile(args.model_config_path, join(save_updir, f'config_{cfg_base_name}.yaml'))
    
    if train_config['if_resume']:
        model.load()
        
    metrics_computer = Kid_Or_Fid(if_cuda=False)
    count = 0
    best_fid_score = 100.
    visualizer.scalars_initialize()
    
    for epoch in range(train_config['epoch']):
        D_f_loss = 0
        D_c_loss = 0
        Gan_loss = 0
        
        for j, variable_list in enumerate(train_loader):
            d_f_loss, d_c_loss, gan_loss = model.one_step(variable_list)
            
            D_f_loss += d_f_loss
            D_c_loss += d_c_loss
            Gan_loss += gan_loss
            
        D_f_loss /= len_along_epoch
        D_c_loss /= len_along_epoch
        Gan_loss /= len_along_epoch
        
        logging.info(
            ">>>>%d: d_f_loss: %5f   d_c_loss: %5f   gan_loss: %5f" % (epoch + 1, D_f_loss, D_c_loss, Gan_loss))
        print(">>>>%d: d_f_loss: %5f   d_c_loss: %5f   gan_loss: %5f" % (epoch + 1, D_f_loss, D_c_loss, Gan_loss))
        visualizer.iter_summarize_performance(None, None, iter(val_loader), str(epoch + 1), combine_trainer=model)
        
        # metrics_computer.update_models(g_fine_model=g_model_fine, g_coarse_model=g_model_coarse)
        # fid_score, kid_mean, kid_std = metrics_computer.spin_once()
        # visualizer.scalar_adjuster([fid_score], epoch+1, 'Fid_score', ['fid_score'])
        # visualizer.scalar_adjuster([kid_mean, kid_std], epoch+1, 'Kid_score', legend=['kid_mean', 'kid_std'])
        visualizer.scalar_adjuster([D_f_loss*10, D_c_loss*10, Gan_loss], epoch+1, "VTGAN_LOSS", 
                                   legend=['df_loss', 'dc_loss', 'gan_loss'])
        count += len_along_epoch
        
        model.save()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='config/train_config.yaml')

    args = parser.parse_args()
    main(args)