import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from baselines.leres.LeReS.Minist_Test.lib.multi_depth_model_woauxi import RelDepthModel
from baselines.leres.LeReS.Minist_Test.lib.net_tools import load_ckpt

from my_python_utils.common_utils import *

from baselines.leres.LeReS.Minist_Test.tools.test_shape import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='/data/vision/torralba/movies_sfm/projects/normals_acc/baselines/leres/LeReS/checkpoints/res101.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


class LeresDepthPredictor(torch.nn.Module):
    def __init__(self, backbone='resnext101'):
        super().__init__()

        self.backbone = backbone
        if self.backbone == 'resnext101':
            checkpoint_file = '/data/vision/torralba/movies_sfm/projects/normals_acc/baselines/leres/LeReS/checkpoints/res101.pth'
        elif self.backbone == 'resnet50':
            checkpoint_file = '/data/vision/torralba/movies_sfm/projects/normals_acc/baselines/leres/LeReS/checkpoints/resnet50.pth'
        else:
            raise Exception("Checkpoint for other backbones needs to be downloaded")

        # create depth model
        self.depth_model = RelDepthModel(backbone=backbone)
        self.depth_model.eval()

        self.shift_model, self.focal_model = make_shift_focallength_models()

        # load checkpoint
        load_ckpt(checkpoint_file, self.depth_model, self.shift_model, self.focal_model)
        self.depth_model.cuda()
        self.shift_model.cuda()
        self.focal_model.cuda()

        self.transform = transforms.Compose([transforms.ToTensor()])


    def compute_depth(self, rgb_c, mask=None):
        C, H, W = rgb_c.shape
        assert C ==  3, "First dim should be color channel"
        assert 0 <= rgb_c.min() and rgb_c.max() <= 1.0, 'Images should be passed normalized between 0 and 1'

        rgb_c_to_leres = np.array(rgb_c * 255.0, dtype='uint8')
        rgb_c_to_leres = rgb_c_to_leres.transpose((1,2,0))

        A_resize = cv2.resize(rgb_c_to_leres, (448, 448))

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = self.depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb_c_to_leres.shape[1], rgb_c_to_leres.shape[0]))

        # recover focal length, shift, and scale-invariant depth
        shift, focal_length, depth_scaleinv = reconstruct3D_from_depth(rgb_c_to_leres, pred_depth_ori, self.shift_model, self.focal_model)

        '''
        # disp = 1 / depth_scaleinv
        # disp = (disp / disp.max() * 60000).astype(np.uint16)

        # if GT depth is available, uncomment the following part to recover the metric depth
        # pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        img_name = v.split('/')[-1]
        # cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        # plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        # cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
        # save disp
        # cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'.png'), disp)

        # reconstruct point cloud from the depth
        pcd = reconstruct_depth(depth_scaleinv, rgb[:, :, ::-1], image_dir_out, img_name[:-4] + '-pcd',
                                focal=focal_length)

        pcd = pcd.transpose(0,1).reshape(rgb.shape)

        show_pointcloud(pcd, rgb)
        '''

        return depth_scaleinv

if __name__ == '__main__':
    args = parse_args()
    leres_depth_predictor = LeresDepthPredictor(backbone=args.backbone)

    image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
    imgs_list = os.listdir(image_dir)
    imgs_list.sort()
    imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
    image_dir_out = image_dir + '/outputs'
    os.makedirs(image_dir_out, exist_ok=True)

    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v)
        rgb = rgb[:, :, ::-1].transpose((2,0,1)).copy()
        predicted_depth = leres_depth_predictor.compute_depth(rgb / 255.0)

        imshow(rgb, title='rgb', biggest_dim=500)
        imshow(predicted_depth, title='predicted_depth', biggest_dim=500)



