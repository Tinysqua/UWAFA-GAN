from dataloader.VT_dataloader import slo_ffa_dataset
import argparse
from utils.common import add_dict_to_argparser, check_dir, convert_to_cuda
from torch.utils import data
from models.modules import Whole_generator
from torchvision.utils import save_image

def main():
    args = create_argparser().parse_args()
    check_dir(args.target_file_path)
    check_dir(args.fake_file_path)
    data_loader = data.DataLoader(slo_ffa_dataset(args.data_path, args.img_size))
    data_iter = iter(data_loader)
    gen = Whole_generator().cuda()
    gen.load_checkpoints(args.model_updir)
    for i in range(args.max_sample):
        var_list = next(data_iter)
        X_realA, X_realB, X_realA_half, X_realB_half = map(convert_to_cuda, var_list)
        X_fakeB_half, X_fakeB = gen.nograd_run(X_realA_half, X_realA)
        save_image(X_realB, f'{args.target_file_path}/{i+1}.png', normalize=True)
        save_image(X_fakeB, f'{args.fake_file_path}/{i+1}.png', normalize=True)

def create_argparser():
    defaults = dict(
        target_file_path='',
        fake_file_path='',
        max_sample=50,
        data_path='dataset/data1', 
        img_size=(832, 1088), 
        model_updir='weights/exp_7_31'
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
      main()