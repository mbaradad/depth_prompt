# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from utils import *
import kornia.geometry

import torch
import torch.nn as nn

from dpt.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

from object_prediction.training_utils import FFTImage, OptimizableImage, rfft2d_freqs
from object_prediction.depth_metrics_and_losses import recover_scale_depth

from torchvision.models import resnet18

import PIL

from collections import OrderedDict
from baselines.boosting_mono_depth.pix2pix.models import networks

from baselines.leres.LeReS.Minist_Test.lib.net_tools import load_ckpt

from torchvision import transforms


class LookAtHomographyMatrix(nn.Module):
  def __init__(self, h, w, debug=False):
    super(LookAtHomographyMatrix, self).__init__()
    self.h, self.w = h, w
    self.debug = debug

  def construct_homography_from_look_at_and_zoom(self, look_at_x, look_at_y, zoom, K):
    cur_device = self.zoom_param.device

    K_inv = torch.inverse(K)

    point_2d = torch.ones(3).to(cur_device)
    point_2d[0] = look_at_x
    point_2d[1] = look_at_y

    point_3d = K_inv @ point_2d.to(cur_device)

    x_rot_rads = torch.atan2(point_3d[1], point_3d[2]) # y / z gives rotation over x
    y_rot_rads = torch.atan2(point_3d[0], point_3d[2]) # x / z gives rotation over y

    x_rot = xrotation_rad_torch(x_rot_rads[None], four_dims=False)[0]
    y_rot = yrotation_rad_torch(y_rot_rads[None], four_dims=False)[0]

    rotation_3D = x_rot @ y_rot
    #if self.debug:
    #  rotated_point = tonumpy(rotation_3D @ point_3d)
    #  rotated_point = rotated_point / np.linalg.norm(rotated_point)
    #   assert np.allclose(rotated_point, (0,0,1), atol=5e-2)

    # K_zoom_centered at c_x_c_y = K_displace_back_to_cx_cy @ K_Zoom_at_origin @ K_displace_cx_cy_to_origin
    # for a zoom of size Z, this is:
    # (Z    0    c_x - Z*c_x)
    # (0    Z    c_y - Z*c_y)
    # (0    0    1)
    '''
    zoom_mat = torch.FloatTensor(((zoom, 0, K[0,2] - zoom * K[0,2]),
                                  (0, zoom, K[1,2] - zoom * K[1,2]),
                                  (0,    0,                      1))).to(cur_device)
    zoom_mat.requires_grad = True

    '''
    zero = torch.zeros(1).to(cur_device)
    one = torch.ones(1).to(cur_device)
    # with torch.cat so that gradients are propagated to zoom param
    if type(zoom) is float:
      zoom = torch.tensor(zoom).to(cur_device)[None]
    zoom_mat = torch.cat([
      torch.cat([zoom, zero, K[0,2] - zoom * K[0,2]], dim=-1)[None,:],
      torch.cat([zero, zoom, K[1,2] - zoom * K[1,2]], dim=-1)[None,:],
      torch.cat([zero, zero,                    one], dim=-1)[None,:]
    ], dim=0)

    # for now without zoom, for debugg purposes
    homography = zoom_mat @ K @ rotation_3D @ K_inv

    return homography

  def get_homography_matrix(self, object_mask, K, batch_index=-1):
    #if self.debug:
    #  zoom = 1
    #else:
    zoom = self.get_zoom(batch_index)
    x, y = self.get_x_y(object_mask, batch_index)

    assert zoom > 0 and x >= 0 and x <= self.w and y >= 0 and y <= self.h

    # similar approach as object_prediction/single_prediction_from_multiviews/image_configs.py, but differentiable
    homography = self.construct_homography_from_look_at_and_zoom(look_at_x=x, look_at_y=y, zoom=zoom, K=K)

    return homography

# the forward populates
class HomographyPredictor(nn.Module):
  def __init__(self, h, w, max_zoom_factor, n_homographies, debug=False):
    super(HomographyPredictor, self).__init__()

    n_hidden = 128

    feature_extractor = resnet18(pretrained=True)
    feature_extractor.fc = nn.Linear(512, n_hidden)
    self.homography_predictor = nn.Sequential(feature_extractor,
                                              nn.ReLU(),
                                              nn.Linear(n_hidden, n_hidden),
                                              nn.ReLU(),
                                              nn.Linear(n_hidden, 3 * n_homographies))
    self.h = h
    self.w = w
    self.max_zoom_factor = max_zoom_factor
    self.predicted_homographies = None
    self.debug = debug

    self.n_homographies = n_homographies

    class PredictedHomography(LookAtHomographyMatrix):
      # TODO: This can be made more efficient, but we do it like this for now (instead of batched end to end)
      # to keep the interface compatible with previous.
      def __init__(self_local, i):
        super(PredictedHomography, self_local).__init__(self.h, self.w, debug=self.debug)
        assert i < self.n_homographies
        self_local.i = i
        # to keep track of device
        self_local.zoom_param = nn.Parameter(torch.normal(mean=torch.zeros(1), std=torch.ones(1)))

      def get_x_y(self_local, object_mask, batch_index=-1):
        assert 0 <= batch_index < len(self.xs)
        return self.xs[batch_index:batch_index+1, self_local.i], self.ys[batch_index:batch_index+1, self_local.i]

      def get_zoom(self_local, batch_index=-1):
        return self.zooms[batch_index:batch_index+1, self_local.i]

    self.homographies = [PredictedHomography(i) for i in range(n_homographies)]

  def get_homography(self, i):
    return self.homographies[i]

  def forward(self, images):
    self.homography_parameters = self.homography_predictor(images)

    self.zooms = 1 + torch.sigmoid(self.homography_parameters[:, 0:self.n_homographies] / 10) * (self.max_zoom_factor - 1)
    self.xs = torch.sigmoid(self.homography_parameters[:, self.n_homographies:2*self.n_homographies] / 10) * self.w
    self.ys = torch.sigmoid(self.homography_parameters[:, 2*self.n_homographies:3*self.n_homographies] / 10) * self.h

class RandomLookAtHomography(LookAtHomographyMatrix):
  def __init__(self, h, w, min_zoom_factor=1, max_zoom_factor=3, debug=False, *args, **kwargs):
    super(RandomLookAtHomography, self).__init__(h, w, debug=debug)
    self.min_zoom_factor = min_zoom_factor
    self.max_zoom_factor = max_zoom_factor

    # to keep track of device, not used
    self.zoom_param = nn.Parameter(torch.normal(mean=torch.zeros(1), std=torch.ones(1)))

  def get_zoom(self, *args, **kwargs):
    return np.random.uniform(self.min_zoom_factor, self.max_zoom_factor)

  def get_x_y(self, object_mask, *args, **kwargs):
    assert len(object_mask.shape) == 2
    # look at a valid position.
    valid_positions = torch.where(object_mask > 0)
    if len(valid_positions[0]) == 0:
      # if no valid positions, just return center
      return self.w // 2, self.h // 2
    else:
      i = np.random.randint(0, len(valid_positions[0]))
    look_at_y, look_at_x = valid_positions[0][i], valid_positions[1][i]
    assert object_mask[look_at_y, look_at_x] > 0
    return look_at_x, look_at_y

class IdentityDifferentiableHomography(LookAtHomographyMatrix):
  def __init__(self, h, w, debug=False, *args, **kwargs):
    super(IdentityDifferentiableHomography, self).__init__(h, w, debug=debug)

    # to keep track of device
    self.zoom_param = nn.Parameter(torch.normal(mean=torch.zeros(1), std=torch.ones(1)))

  def get_zoom(self, *args, **kwargs):
    return 1

  def get_x_y(self, *args, **kwargs):
    x = 0.5 * self.w
    y = 0.5 * self.h
    return x, y

  def get_homography_matrix(self, object_mask, K, batch_index=-1):
    cur_device = self.zoom_param.device
    homography = torch.eye(3)
    homography = homography.to(cur_device)

    return homography

class DifferentiableLookAtHomography(LookAtHomographyMatrix):
  def __init__(self, h, w, min_zoom_factor=1, max_zoom_factor=3, debug=False):
    super(DifferentiableLookAtHomography, self).__init__(h, w, debug)

    self.min_zoom_factor = min_zoom_factor
    self.max_zoom_factor = max_zoom_factor

    self.zoom_param = nn.Parameter(torch.normal(mean=torch.zeros(1),
                                                std=torch.ones(1)))
    self.canvas_positions_params = nn.Parameter(torch.FloatTensor(torch.normal(mean=torch.zeros(2), std=torch.ones(2))))

  def get_zoom(self, *args, **kwargs):
    # divide by a constant so that it does not change that abruptly
    return (1 + torch.sigmoid(self.zoom_param / 10) * self.max_zoom_factor - 1)

  def get_x_y(self, *args, **kwargs):
    # divide by a constant so that it does not change that abruptly
    x = torch.sigmoid(self.canvas_positions_params[0] / 10) * self.w
    y = torch.sigmoid(self.canvas_positions_params[1] / 10) * self.h
    return x, y

class UnetHyperNetwork(nn.Module):
  def __init__(self, distortion_img_type, img_size, add_bias_to_unet=True, unet_input='mask'):
    super(UnetHyperNetwork, self).__init__()
    self.distortion_img_type = distortion_img_type
    assert distortion_img_type in ['fft', 'image'], "Distortion img type {} not available!".format(distortion_img_type)
    assert unet_input in ['image', 'mask']
    self.img_size = img_size
    w = self.img_size
    h = self.img_size

    n_inputs = 1 if distortion_img_type == 'image' else 2
    n_inputs *= 1 if unet_input == 'mask' else 3
    n_outputs = 3 if distortion_img_type == 'image' else 6

    if img_size == 384 or img_size == 448:
      # unet_384 seems to fail, as downsampling for inner layer is too big
      # it seems that for any resolution in [512, 384, 256] the definition is exactly the same as for 512
      self.model = networks.define_G(n_inputs, n_outputs, 64, 'unet_128', 'none', False, 'normal', 0.02, [])
    elif img_size == 512:
      self.model = networks.define_G(n_inputs, n_outputs, 64, 'unet_512', 'none', False, 'normal', 0.02, [])
    else:
      raise Exception("Unet needs to be adapted to resolution {}".format(img_size))

    # remove tanh, as we normalize afterwards with The FFTImage/OptimizableImage
    self.model.model.model[-1] = nn.Identity()

    self.add_bias_to_unet = add_bias_to_unet
    if self.add_bias_to_unet:
      if distortion_img_type == 'fft':
        self.optimized_distortion = FFTImage(w=w, h=h)
      else:
        self.optimized_distortion = OptimizableImage(w=w, h=h)


  def get_parameters(self):
    model_parameters = list(self.model.parameters())
    if self.add_bias_to_unet:
      model_parameters.extend(list(self.optimized_distortion.parameters()))
    return model_parameters

  def forward(self, inputs):
    if self.img_size == 448:
      # for leres network,
      inputs = F.interpolate(inputs, size=(384, 384), mode="bilinear")

    if self.distortion_img_type == 'image':
      conditional_distortion_params = self.model(inputs)
      # postprocess so that stays in (0,1) range
      conditional_distortion = nn.Sigmoid()(conditional_distortion_params * 10)
    else:
      _, _, h, w = inputs.shape

      spectrum = torch.fft.rfftn(inputs, s=(h, w), norm='ortho')
      spectrum = torch.cat((spectrum.real, spectrum.imag), 1)
      assert spectrum.shape[2:] == (384, 193)

      input_spectrum = torch.nn.functional.pad(spectrum, (0,384 - 193), mode='constant', value=0)

      output_spectrum = self.model(input_spectrum)[:,:,:,:193].reshape((-1, 2, 3, 384, 193))

      output_spectrum = torch.complex(output_spectrum[:,0,:,:193],output_spectrum[:,1,:,:193])

      image_before_norm = torch.fft.irfftn(output_spectrum, s=(h, w), norm='ortho')
      conditional_distortion = nn.Sigmoid()(image_before_norm * 40)

      # output_spectrum = torch.complex(input_spectrum[:,0,:,:193],input_spectrum[:,1,:,:193])
      # to scale as 1/f, if initialization is bad
      # freqs = rfft2d_freqs(h, w)
      # decay_power = 1
      # scale = totorch(1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power).cuda()
      # scaled_spectrum_t = scale * spectrum


    if self.img_size == 448:
      # resize if necessary
      conditional_distortion = F.interpolate(conditional_distortion, size=(448, 448), mode="bilinear")


    # add a constant directly to prediciton, and average
    if self.add_bias_to_unet:
      bs = inputs.shape[0]
      bias_image = self.optimized_distortion.get_image().repeat((bs,1,1,1))
      return 0.5 * (conditional_distortion + bias_image)
    else:
      return conditional_distortion

class ModelWithExtras(nn.Module):
  def __init__(self, model,
               hyper_optimization,
               add_bias_to_unet,
               distortion_img_type,
               distortion_mode,
               h, w,
               debug=False,
               finetune_network=False,
               unet_input='mask',
               image_range_after_transform=None):
    super(ModelWithExtras, self).__init__()
    self.model = model

    assert hyper_optimization in ['none', 'image', 'unet']
    assert distortion_img_type in ['fft', 'image']
    assert hyper_optimization == 'none' or not finetune_network, "Finetuning should only be active if hyper_optimization is none!"
    assert unet_input in ['mask', 'image'], "Unet input should be mask or image for ablation."

    assert image_range_after_transform is not None, "Image range after transform should be passed as argument"
    self.image_range_after_transform = image_range_after_transform

    self.h = h
    self.w = w

    self.hyper_optimization = hyper_optimization
    self.background_img_type = distortion_img_type
    self.distortion_mode = distortion_mode

    self.debug = debug
    self.finetune_network = finetune_network
    self.unet_input = unet_input

    if self.hyper_optimization == 'image':
      if distortion_img_type == 'fft':
        self.optimized_distortion = FFTImage(w=w, h=h)
      else:
        self.optimized_distortion = OptimizableImage(w=w, h=h)

    elif self.hyper_optimization == 'unet':
      assert w == h
      self.hyper_network = UnetHyperNetwork(distortion_img_type,
                                            img_size=w,
                                            add_bias_to_unet=add_bias_to_unet,
                                            unet_input=unet_input)

  def get_trainable_parameters(self):
    params = []
    if self.hyper_optimization == 'image':
      params.extend(self.optimized_distortion.get_parameters())
    elif self.hyper_optimization == 'unet':
      params.extend(self.hyper_network.get_parameters())
    if self.finetune_network:
      params.extend(self.model.parameters())
    assert len(params) > 0, "No parameters to be optimized on this model"
    return params

  def get_background_image(self, inputs, object_masks):
    if self.hyper_optimization == 'image':
      background = self.optimized_distortion.get_image()

      bs = inputs.shape[0]
      backgrounds = background.repeat(bs, 1, 1, 1)
    elif self.hyper_optimization == 'unet':
      if self.unet_input == 'image':
        backgrounds = self.hyper_network(inputs.cuda())
      else:
        backgrounds = self.hyper_network(object_masks.cuda())

    assert backgrounds.min() >= 0 and backgrounds.max() <= 1, "Background image before range adjustment should be between 0 and 1"
    backgrounds = self.image_range_after_transform[0] + backgrounds * (self.image_range_after_transform[1] - self.image_range_after_transform[0])

    return backgrounds

  def get_images_with_background(self, inputs, object_masks, return_backgrounds=False):
    assert self.hyper_optimization != 'none', "Can only be called when hyper_optimization != 'none'"
    assert self.distortion_mode in ['additive', 'background']
    assert not object_masks is None, "Objects mask should be passed when optimizing homographies!"
    assert len(object_masks.shape) == len(inputs.shape) == 4
    bg_images = self.get_background_image(inputs, object_masks)

    assert bg_images.shape == inputs.shape
    if self.distortion_mode == 'background':
      inputs = inputs * object_masks + (1 - object_masks) * bg_images
    elif self.distortion_mode == 'additive':
      inputs = inputs + bg_images
    else:
      raise Exception("Background mode {} not implemented".format(self.distortion_mode))
    if return_backgrounds:
      return inputs, bg_images
    else:
      return inputs

  # returns n_disparities per element in the batch, taking into accout number of homographies N:
  # returned disparities = BxNxHxW
  def forward(self, inputs, object_masks=None):
    debug_info = dict()
    inputs_to_network = inputs

    if self.hyper_optimization != 'none':
      inputs_to_network, backgrounds = self.get_images_with_background(inputs_to_network, object_masks[:,None,:,:], return_backgrounds=True)
      debug_info['bg_image'] = tonumpy(backgrounds)

    debug_info['model_inputs'] = tonumpy(inputs_to_network)
    depth = self.model(inputs_to_network)

    debug_info['depth'] = tonumpy(depth)

    return depth, debug_info

  # to keep the API signature with the other depth predictions, but it's the same as get_depth in practice
  def compute_depth_batched(self, inputs, masks=None):
    return self.forward(inputs, masks)

  def get_depth(self, inputs, object_masks=None):
    return self.forward(inputs, object_masks)

def get_dpt_transform(model_type):
  assert model_type in ['DPT_Large', 'DPT_Hybrid', 'MiDaS', 'MiDaS_small']
  net_w = net_h = 384
  if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    resize_mode = "minimal"
  else:
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    resize_mode = "upper_bound"
    normalization = NormalizeImage(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

  img_transform = Compose(
    [
      Resize(
        net_w,
        net_h,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method=resize_mode,
        image_interpolation_method=cv2.INTER_CUBIC,
      ),
      normalization,
      PrepareForNet(),
    ]
  )

  return img_transform

class MidasWrapper(nn.Module):
  def __init__(self, model):
    super(MidasWrapper, self).__init__()
    self.midas_model = model

  def forward(self, inputs):
    midas_output = self.midas_model.forward(inputs)

    # applies the same transformation to get depth from disparity as the heuristic in facebook_photo3d
    # https://github.com/vt-vl-lab/3d-photo-inpainting
    disp = midas_output
    disp = disp - disp.min(-1)[0].min(-1)[0][:,None,None]
    # disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
    disp = (disp / disp.max(-1)[0].max(-1)[0][:,None,None])
    depth = 1. / torch.clip(disp, min=0.05)

    return depth

def get_model_and_transform(args):
  if args.model.startswith('dpt') or args.model.startswith('midas'):
    if args.model == 'dpt':
      model_type = 'DPT_Large'
    elif args.model == 'dpt_hybrid':
      model_type = 'DPT_Hybrid'
    elif args.model == 'midas_conv':
      model_type = 'MiDaS'
    elif args.model == 'midas_conv_small':
      model_type = 'MiDaS_small'

    model = torch.hub.load("intel-isl/MiDaS", model_type)
    image_range_after_transform = [-1, 1]
    image_size = 384

    model = MidasWrapper(model)
    img_transform = get_dpt_transform(model_type)

  elif args.model == 'omnidata':
    # replicating
    image_size = 384

    root_dir = '/data/vision/torralba/movies_sfm/home/normals_acc/baselines/omnidata_main/omnidata_tools/torch/pretrained_models'
    pretrained_weights_path = root_dir + '/omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'

    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    from baselines.omnidata_main.omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
    if 'state_dict' in checkpoint:
      state_dict = {}
      for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
    else:
      state_dict = checkpoint
    model.load_state_dict(state_dict)

    img_transform = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

    image_range_after_transform = [-1, 1]

  elif args.model.startswith('leres'):
    # from https://github.com/aim-uofa/AdelaiDepth/blob/main/LeReS/Minist_Test/tools/test_depth.py

    backbone = args.model.split('_')[1]
    available_backbones = ['resnet50', 'resnext101']
    assert backbone in available_backbones, 'Backbone {} not implemented for leres, should be one of {}'.format(backbone, available_backbones)

    image_size = 448

    img_transform = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                             (0.229, 0.224, 0.225))])

    image_range_after_transform = [-3, 3]

    from baselines.leres.LeReS.Minist_Test.lib.multi_depth_model_woauxi import DepthModel

    class DepthModelWithForward(DepthModel):
      def forward(self, rgb):
        pred = super().forward(rgb)[:,0]
        pred_depth_out = pred - pred.min(-1)[0].min(-1)[0][:,None,None] + 0.01
        return pred_depth_out

    if backbone == 'resnet50':
        encoder_type = 'resnet50_stride32'
    elif backbone == 'resnext101':
        encoder_type = 'resnext101_stride32x8d'

    model = DepthModelWithForward(encoder_type)

    # load checkpoint
    checkpoint_file = '/data/vision/torralba/movies_sfm/home/normals_acc/baselines/leres/LeReS/checkpoints/{}.pth'.format(backbone)
    print("loading checkpoint %s" % checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    def strip_prefix_if_present(state_dict, prefix):
      keys = sorted(state_dict.keys())
      if not all(key.startswith(prefix) for key in keys):
        return state_dict
      stripped_state_dict = OrderedDict()
      for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
      return stripped_state_dict

    model.load_state_dict(strip_prefix_if_present(strip_prefix_if_present(checkpoint['depth_model'], "module."), 'depth_model.'), strict=True)
    del checkpoint
    torch.cuda.empty_cache()

  else:
    raise Exception("Model {} not available".format(args.model))

  if not args.pretrained:
    # reset weights, solution from: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
    def weight_reset(m):
      # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

    model.apply(weight_reset)

  if not args.pretrained:
    # put on eval mode
    model = model.eval()

  model = ModelWithExtras(model,
                          h=image_size, w=image_size,

                          image_range_after_transform=image_range_after_transform,

                          hyper_optimization=args.hyper_optimization,
                          unet_input=args.unet_input,
                          add_bias_to_unet=args.add_bias_to_unet,

                          distortion_img_type=args.distortion_img_type,
                          distortion_mode=args.distortion_mode,
                          debug=args.debug,

                          finetune_network=args.finetune_network)

  return model, img_transform

# same as MidasDispComputer but returns up-to-scale depth directly, as it is trained on up-to-scale depth.
class DPTDepthComputer():
  def __init__(self, checkpoint_file, model_type='dpt'):
    checkpoint = torch.load(checkpoint_file)
    state_dict = dict([(k.replace('module.', ''), v) for k,v in checkpoint['state_dict'].items()])

    if len([k for k in state_dict.keys() if 'background_image' in k]) >= 1:
      self.hyper_optimization = 'image'
    elif len([k for k in state_dict.keys() if 'hyper_network' in k]) >= 1:
      self.hyper_optimization = 'unet'
    else:
      self.hyper_optimization == 'none'

    args = AttrDict({'model': model_type,
                     'hyper_optimization': self.hyper_optimization,
                     'pretrained': True,
                     'debug': False
    })

    self.model, self.transform = get_model_and_transform(args)
    self.model.load_state_dict(state_dict)

    self.model = self.model.cuda()
    self.model.eval()

  def compute_depth_batched(self, imgs, masks=None):
    masks = masks.cuda()
    input_batch = imgs.cuda()

    original_masks = masks

    with no_context():
    # TODO: torch.no_grad() was using more memory than with grad, although is not an issue
    #with torch.no_grad():
      if self.optimize_background:
        if masks == 'None':
          raise Exception("Mask should be passed when optimizing background is set to True")
        assert original_masks.shape[-2] / original_masks.shape[-1] == original_masks.shape[-2] / original_masks.shape[-1]

      prediction = self.model(input_batch, original_masks)
      prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=imgs.shape[-2:],
        mode="bicubic",
        align_corners=False,
      ).squeeze()

    dpt_depth = prediction.detach()

    assert dpt_depth.shape[1:] == imgs.shape[-2:], "Shapes don't match"

    return dpt_depth

  def compute_depth(self, img_file_or_img, device='cuda:0', biggest_image_dim=-1, mask=None):
    if type(img_file_or_img) is str:
      img = cv2.imread(img_file_or_img)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      assert img_file_or_img.shape[0] == 3, "Should be passed with first dimension being channel, img.shape[0] == 3"
      img = img_file_or_img.transpose((1, 2, 0))

    if biggest_image_dim != -1:
      img = scale_image_biggest_dim(img.transpose((2, 0, 1)), biggest_image_dim).transpose((1, 2, 0))

    input_batch = totorch(self.transform(({'image': img / 255.0}))['image'])[None]
    if not mask is None:
      mask = myimresize(mask * 1.0, target_shape=(input_batch.shape[2:]))

    return self.compute_depth_batched(input_batch, None if mask is None else mask[None]).cpu().numpy()


if __name__ == '__main__':
  net = UnetHyperNetwork('image',
                   img_size=384,
                   add_bias_to_unet=False)
  print(count_trainable_parameters(net))