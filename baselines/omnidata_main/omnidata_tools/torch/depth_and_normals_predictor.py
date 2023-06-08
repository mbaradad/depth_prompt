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

from baselines.omnidata_main.omnidata_tools.torch.modules.unet import UNet
from baselines.omnidata_main.omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
from baselines.omnidata_main.omnidata_tools.torch.data.transforms import get_transform

from my_python_utils.common_utils import *

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


class OmnidataDepthAndNormalsPredictor(torch.nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task

        # self.trans_topil = transforms.ToPILImage()
        # os.system(f"mkdir -p {args.output_path}")

        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        root_dir = '/data/vision/torralba/movies_sfm/projects/normals_acc/baselines/omnidata_main/omnidata_tools/torch/pretrained_models/'

        # get target task and model
        if self.task == 'normal':
            self.image_size = 384

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
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
            checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k[6:]] = v
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.transform = transforms.Compose([transforms.Resize(self.image_size),
                                                 transforms.CenterCrop(self.image_size),
                                                 get_transform('rgb', image_size=None)])


        elif self.task == 'depth':
            self.image_size = 384
            pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
            # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
            self.model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
            checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k[6:]] = v
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.transform = transforms.Compose([transforms.Resize(self.image_size),
                                                 transforms.CenterCrop(self.image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=0.5, std=0.5)])

        else:
            print("task should be one of the following: normal, depth")
            sys.exit()

        self.trans_rgb = transforms.Compose([transforms.Resize(512),
                                        transforms.CenterCrop(512)])

    def compute(self, rgb_c):
        C, H, W = rgb_c.shape
        assert C ==  3, "First dim should be color channel"
        assert 0 <= rgb_c.min() and rgb_c.max() <= 1.0, 'Images should be passed normalized between 0 and 1'

        with torch.no_grad():
            img = PIL.Image.fromarray(np.array(rgb_c.transpose((1,2,0)) * 255.0, dtype='uint8'))
            img_tensor = self.transform(img)[:3].unsqueeze(0).to(self.device)

            if img_tensor.shape[1] == 1:
                # BW
                img_tensor = img_tensor.repeat_interleave(3, 1)

            output = self.model(img_tensor).clamp(min=0, max=1)

            if self.task == 'depth':
                output = F.interpolate(output.unsqueeze(0), (H,W), mode='bicubic').squeeze(0).squeeze(0)
            else:
                output = F.interpolate(output, (H, W), mode='nearest').squeeze(0)

                output_to_swap = tonumpy(output) * 2 - 1
                output_resorted = np.array(output_to_swap)
                output_resorted[1] = -1 * output_to_swap[1]
                output_resorted[0] = output_to_swap[0]
                output_resorted[2] = -1 * output_to_swap[2]

                output = output_resorted

            return tonumpy(output)

    def compute_depth_batched(self, imgs, masks=None):
        _, C, H, W = imgs.shape

        output = self.model(imgs).clamp(min=0, max=1)
        output = F.interpolate(output[:,None], (H, W), mode='bicubic')[:,0]

        return output

    def compute_depth(self, rgb_c, masks=None):
        assert self.task == 'depth', "Predictor not initialized with task == depth"
        return self.compute(rgb_c)

    def compute_normals(self, rgb_c, masks=None):
        assert self.task == 'normal', "Predictor not initialized with task == normal"
        return self.compute(rgb_c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

    parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
    parser.set_defaults(img_path='/data/vision/torralba/movies_sfm/projects/normals_acc/data/test_images/google_scans/Schleich_Spinosaurus_Action_Figure_texture_04.jpg')

    parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
    parser.set_defaults(output_path='/tmp/omnidata_tests')

    args = parser.parse_args()

    trans_topil = transforms.ToPILImage()

    os.system(f"mkdir -p {args.output_path}")
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    only_show = True
    img_path = Path(args.img_path)

    depth_predictor = OmnidataDepthAndNormalsPredictor('depth')
    normals_predictor = OmnidataDepthAndNormalsPredictor('normal')

    img = cv2_imread(args.img_path) / 255.0

    C, H, W = img.shape

    depth = depth_predictor.compute_depth(img)
    pred_normals = normals_predictor.compute_normals(img)

    K = np.array([[886.81 / 1024 * W,   0.  , W / 2  ],
                  [  0.  , 886.81 / 1024 * W, H / 2  ],
                  [  0.  ,   0.  ,   1.  ]], dtype='float32')

    pcl = pixel2cam(totorch(depth).cpu(), totorch(K)[None])[0]
    normals_from_depth = compute_normals_from_closest_image_coords(pcl[None])[0]

    imshow(pred_normals, title='normals_from_model')
    imshow(depth, title='depth')
    imshow(normals_from_depth, title='normals_from_depth')

    show_pointcloud(pcl, np.array(img * 255.0, dtype='uint8'), title='predicted_pcl')