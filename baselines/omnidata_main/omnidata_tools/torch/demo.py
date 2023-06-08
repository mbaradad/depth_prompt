import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

from my_python_utils.common_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

    parser.add_argument('--task', dest='task', help="normal or depth")
    parser.set_defaults(task='normal')

    parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
    parser.set_defaults(img_path='/data/vision/torralba/movies_sfm/projects/normals_acc/data/hypersim/ai_001_010_images_scene_cam_00_final_hdf5_frame.0000.color.hdf5.png')

    parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
    parser.set_defaults(output_path='/tmp/omnidata_tests')

    args = parser.parse_args()

    root_dir = '/data/vision/torralba/movies_sfm/projects/normals_acc/baselines/omnidata_main/omnidata_tools/torch/pretrained_models/'

    trans_topil = transforms.ToPILImage()

    os.system(f"mkdir -p {args.output_path}")
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # get target task and model
    if args.task == 'normal':
        image_size = 384

        ## Version 1 model
        # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
        # model = UNet(in_channels=3, out_channels=3)
        # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

        # if 'state_dict' in checkpoint:
        #     state_dict = {}
        #     for k, v in checkpoint['state_dict'].items():
        #         state_dict[k.replace('model.', '')] = v
        # else:
        #     state_dict = checkpoint


        pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            get_transform('rgb', image_size=None)])

    elif args.task == 'depth':
        image_size = 384
        pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5)])

    else:
        print("task should be one of the following: normal, depth")
        sys.exit()

    trans_rgb = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(512)])


    def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
        if mask_valid is not None:
            img[~mask_valid] = torch.nan
        sorted_img = torch.sort(torch.flatten(img))[0]
        # Remove nan, nan at the end of sort
        num_nan = sorted_img.isnan().sum()
        if num_nan > 0:
            sorted_img = sorted_img[:-num_nan]
        # Remove outliers
        trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
        trunc_mean = trunc_img.mean()
        trunc_var = trunc_img.var()
        eps = 1e-6
        # Replace nan by mean
        img = torch.nan_to_num(img, nan=trunc_mean)
        # Standardize
        img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
        return img


    def save_outputs(img_path, output_file_name, only_show=True):
        with torch.no_grad():
            save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')

            print(f'Reading input {img_path} ...')
            img = Image.open(img_path)

            img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

            rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
            trans_rgb(img).save(rgb_path)

            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3,1)

            output = tonumpy(model(img_tensor).clamp(min=0, max=1))

            if args.task == 'normal':
                # adapt to match convention that we use in other parts.
                output_to_swap = output[0] * 2 - 1
                output_resorted = np.array(output_to_swap)
                output_resorted[1] = -1 * output_to_swap[1]
                output_resorted[0] = output_to_swap[0]
                output_resorted[2] = -1 * output_to_swap[2]
                imshow(output_resorted, normalize_image=True)

            if only_show:
                imshow(output)
                return

            if args.task == 'depth':
                output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
                output = output.clamp(0,1)
                output = 1 - output
    #             output = standardize_depth_map(output)
                plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')

            else:
                trans_topil(output[0]).save(save_path)

            print(f'Writing output {save_path} ...')

    only_show = True

    img_path = Path(args.img_path)
    if img_path.is_file():
        save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0], only_show=only_show)
    elif img_path.is_dir():
        for f in glob.glob(args.img_path+'/*'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], only_show=only_show)
    else:
        print("invalid file path!")
        sys.exit()