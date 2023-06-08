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


import numpy as np
import torch
import tqdm
import os

import skvideo
import sys
import os
import imageio

import math
import cv2

from kornia.geometry.camera.pinhole import pixel2cam as k_pixel2cam
from pathlib import Path
import argparse

import random

from PIL import Image
import contextlib

import torch.nn.functional as F

if 'anaconda' in sys.executable:
  # set ffmpeg to anaconda path
  skvideo.setFFmpegPath(os.path.split(sys.executable)[0])
else:
  skvideo.setFFmpegPath('/usr/bin')
from skvideo.io import FFmpegWriter, FFmpegReader

class MyVideoWriter():
  def __init__(self, file, fps=None, verbosity=0, *args, **kwargs):
    if not fps is None:
      kwargs['inputdict'] = {'-r': str(fps)}
    kwargs['verbosity'] = verbosity
    assert verbosity in range(2), "Verbosity should be between 0 or 1"
    self.video_writer = FFmpegWriter(file, *args, **kwargs)

  def writeFrame(self, im):
    if len(im.shape) == 3 and im.shape[0] == 3:
      transformed_image = im.transpose((1,2,0))
    elif len(im.shape) == 2:
      transformed_image = np.concatenate((im[:,:,None], im[:,:,None], im[:,:,None]), axis=-1)
    else:
      transformed_image = im
    self.video_writer.writeFrame(transformed_image)

  def close(self):
    self.video_writer.close()

class MyVideoReader():
  def __init__(self, video_file):
    if video_file.endswith('.m4v'):
      self.vid = imageio.get_reader(video_file, format='.mp4')
    else:
      self.vid = imageio.get_reader(video_file)
    self.frame_i = 0

  def get_next_frame(self):
    try:
      return np.array(self.vid.get_next_data().transpose((2,0,1)))
    except:
      return None

  def get_n_frames(self):
    return int(math.floor(self.get_duration_seconds() * self.get_fps()))

  def get_duration_seconds(self):
    return self.vid._meta['duration']

  def get_fps(self):
    return self.vid._meta['fps']

  def position_cursor_frame(self, i):
    assert i < self.get_n_frames()
    self.frame_i = i
    self.vid.set_image_index(self.frame_i)

  def get_frame_i(self, i):
    old_frame_i = self.frame_i
    self.position_cursor_frame(i)
    frame = self.get_next_frame()
    self.position_cursor_frame(old_frame_i)
    return frame

  def is_opened(self):
    return not self.vid.closed

def listdir(folder, prepend_folder=False, extension=None, type=None):
  assert type in [None, 'file', 'folder'], "Type must be None, 'file' or 'folder'"
  files = [k for k in os.listdir(folder) if (True if extension is None else k.endswith(extension))]
  if type == 'folder':
    files = [k for k in files if os.path.isdir(folder + '/' + k)]
  elif type == 'file':
    files = [k for k in files if not os.path.isdir(folder + '/' + k)]
  if prepend_folder:
    files = [folder + '/' + f for f in files]
  return files


def compute_normals_from_closest_image_coords(coords, mask=None):
  # TODO: maybe change cs to be [..., :-1,:-1] as it seems more intuitive
  assert coords.shape[1] == 3 and len(coords.shape) == 4
  assert mask is None or len(mask.shape) == 4
  assert mask is None or (mask.shape[0] == coords.shape[0] and mask.shape[2:] == coords.shape[2:])

  x_coords = coords[:,0,:,:]
  y_coords = coords[:,1,:,:]
  z_coords = coords[:,2,:,:]

  if type(coords) is torch.Tensor or type(coords) is torch.nn.parameter.Parameter:
    ts = torch.cat((x_coords[:, None, :-1, 1:], y_coords[:, None, :-1, 1:], z_coords[:, None,:-1,1:]), dim=1)
    ls = torch.cat((x_coords[:, None, 1:, :-1], y_coords[:, None, 1:, :-1], z_coords[:, None, 1:, :-1]), dim=1)
    cs = torch.cat((x_coords[:, None, 1:, 1:], y_coords[:, None, 1:, 1:], z_coords[:, None,1:,1:]), dim=1)

    n = torch.cross((ls - cs),(ts - cs), dim=1) * 1e10

    # if normals appear incorrect, it may be becuase of the 1e-20, if the scale of the pcl is too small,
    # it was giving errors with a constant of 1e-5 (and it was replaced to 1e-20). We also need an epsilon to avoid nans when using this function for training.
    n_norm = n/(torch.sqrt(torch.abs((n*n).sum(1) + 1e-20))[:,None,:,:])
  else:
    ts = np.concatenate((x_coords[:, None, :-1, 1:], y_coords[:, None, :-1, 1:], z_coords[:, None,:-1,1:]), axis=1)
    ls = np.concatenate((x_coords[:, None, 1:, :-1], y_coords[:, None, 1:, :-1], z_coords[:, None, 1:, :-1]), axis=1)
    cs = np.concatenate((x_coords[:, None, 1:, 1:], y_coords[:, None, 1:, 1:], z_coords[:, None,1:,1:]), axis=1)

    n = np.cross((ls - cs),(ts - cs), axis=1)
    n_norm = n/(np.sqrt(np.abs((n*n).sum(1) + 1e-20))[:,None,:,:])

  if not mask is None:
    valid_ts = mask[:,:, :-1, 1:]
    valid_ls = mask[:,:, 1:, :-1]
    valid_cs = mask[:,:, 1:, 1:]
    if type(mask) is torch.Tensor:
      final_mask = valid_ts * valid_ls * valid_cs
    else:
      final_mask = np.logical_and(np.logical_and(valid_ts, valid_ls), valid_cs)
    return n_norm, final_mask
  else:
    return n_norm

pixel_coords_cpu = None
pixel_coords_cuda = None

def get_id_grid(height, width):
  return create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2

def set_id_grid(height, width, to_cuda=False):
  global pixel_coords_cpu, pixel_coords_cuda

  pixel_grid = get_id_grid(height, width)
  if to_cuda:
    pixel_coords_cuda = pixel_grid.cuda()
  else:
    pixel_coords_cpu = pixel_grid.cpu()

from kornia.geometry.camera.pinhole import pixel2cam as k_pixel2cam
from kornia.utils import create_meshgrid
from kornia.geometry.conversions import convert_points_to_homogeneous

def get_id_grid(height, width):
  grid = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
  return convert_points_to_homogeneous(grid)


def make_4x4_K(K):
  batch_size = K.shape[0]
  zeros = torch.zeros((batch_size,3,1))
  with_1 = torch.Tensor(np.array((0,0,0,1)))[None,:None:].expand(batch_size,1,4)
  if K.is_cuda:
    zeros = zeros.cuda()
    with_1 = with_1.cuda()
  K = torch.cat((K, zeros), axis=2)
  K = torch.cat((K, with_1), axis=1)

  return K


def pixel2cam(depth, K):
  global pixel_coords_cpu, pixel_coords_cuda
  if len(depth.shape) == 4:
    assert depth.shape[1] == 1
    depth = depth[1]
  assert len(depth.shape) == 3
  assert K.shape[1] == K.shape[2]
  assert depth.shape[0] == K.shape[0]

  K = make_4x4_K(K)
  intrinsics_inv = torch.inverse(K)

  height, width = depth.shape[-2:]
  if depth.is_cuda:
    # to avoid recomputing the id_grid if it is not necessary
    if (pixel_coords_cuda is None) or pixel_coords_cuda.size(2) != height or pixel_coords_cuda.size(3) != width:
      set_id_grid(height, width, to_cuda=True)
    pixel_coords = pixel_coords_cuda
  else:
    if (pixel_coords_cpu is None) or pixel_coords_cpu.size(2) != height or pixel_coords_cpu.size(3) != width:
      set_id_grid(height, width, to_cuda=False)
    pixel_coords = pixel_coords_cpu

  batch_size = depth.shape[0]
  pcl = k_pixel2cam(depth[:,None,:,:], intrinsics_inv, pixel_coords.expand(batch_size, -1, -1, -1))
  return pcl.permute(0,3,1,2)


def totorch(array, device=None):
  if type(array) is torch.Tensor:
    return array
  if not type(array) is np.ndarray:
    array = np.array(array)
  array = torch.FloatTensor(array)
  if not device is None:
    array = array.to(device)
  return array


def tonumpy(tensor):
  if type(tensor) is Image:
    return np.array(tensor).transpose((2,0,1))
  if type(tensor) is list:
    return np.array(tensor)
  if type(tensor) is np.ndarray:
    return tensor
  if tensor.requires_grad:
    tensor = tensor.detach()
  if type(tensor) is torch.autograd.Variable:
    tensor = tensor.data
  if tensor.is_cuda:
    tensor = tensor.cpu()
  return tensor.detach().numpy()


def cv2_imread(file, return_BGR=False, read_alpha=False):
  im = None
  if read_alpha:
    try:
      im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    except:
      print("Failed to read alpha channel, will us standard imread!")
  if not read_alpha or im is None:
    im = cv2.imread(file)
  if im is None:
    raise Exception('Image {} could not be read!'.format(file))
  im = im.transpose(2,0,1)
  if return_BGR:
    return im
  if im.shape[0] == 4:
    return np.concatenate((im[:3][::-1], im[3:4]))
  else:
    return im[::-1, :, :]


def read_text_file_lines(filename, stop_at=-1):
  lines = list()
  with open(filename, 'r') as f:
    for line in f:
      if stop_at > 0 and len(lines) >= stop_at:
        return lines
      lines.append(line.replace('\n',''))
  return lines

def write_text_file_lines(lines, file):
  assert type(lines) is list, "Lines should be a list of strings"
  with open(file, 'w') as file_handler:
    for item in lines:
      file_handler.write("%s\n" % item)

def write_text_file(text, filename):
  with open(filename, "w") as file:
    file.write(text)

def read_text_file(filename):
  text_file = open(filename, "r")
  data = text_file.read()
  text_file.close()
  return data



def crop_center(img, crop):
  cropy, cropx = crop
  if len(img.shape) == 3:
    _, y, x = img.shape
  else:
    y, x = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  if len(img.shape) == 3:
    return img[:, starty:starty + cropy, startx:startx + cropx]
  else:
    return img[starty:starty + cropy, startx:startx + cropx]

def cv2_resize(image, target_height_width, interpolation=cv2.INTER_NEAREST):
  if len(image.shape) == 2:
    return cv2.resize(image, target_height_width[::-1], interpolation=interpolation)
  else:
    return cv2.resize(image.transpose((1, 2, 0)), target_height_width[::-1], interpolation=interpolation).transpose((2, 0, 1))

def get_image_x_y_coord_map(height, width):
  # returns image coord maps from x_y, from 0 to width -1, 0 to
  return

def best_centercrop_image(image, height, width, return_rescaled_size=False, interpolation=cv2.INTER_NEAREST):
  if height == -1 and width == -1:
    if return_rescaled_size:
      return image, image.shape
    return image
  image_height, image_width = image.shape[-2:]
  im_crop_height_shape = (int(height), int(image_width * height / image_height))
  im_crop_width_shape = (int(image_height * width / image_width), int(width))
  # if we crop on the height dimension, there must be enough pixels on the width
  if im_crop_height_shape[1] >= width:
    rescaled_size = im_crop_height_shape
  else:
    # crop over width
    rescaled_size = im_crop_width_shape
  resized_image = cv2_resize(image, rescaled_size, interpolation=interpolation)
  center_cropped = crop_center(resized_image, (height, width))
  if return_rescaled_size:
    return center_cropped, rescaled_size
  else:
    return center_cropped


def select_gpus(gpus_arg):
  #so that default gpu is one of the selected, instead of 0
  gpus_arg = str(gpus_arg)
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = list(range(len(gpus_arg.split(','))))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

  flag = 0
  for i in range(len(gpus)):
    for i1 in range(len(gpus)):
      if i != i1:
        if gpus[i] == gpus[i1]:
          flag = 1
  assert not flag, "Gpus repeated: {}".format(gpus)

  return gpus


def list_of_lists_into_single_list(list_of_lists):
  flat_list = [item for sublist in list_of_lists for item in sublist]
  return flat_list


def find_all_files_recursively(folder, prepend_folder=False, extension=None, progress=False, substring=None, include_folders=False, max_n_files=-1):
  if extension is None:
    glob_expresion = '*'
  else:
    glob_expresion = '*' + extension
  all_files = []
  for f in Path(folder).rglob(glob_expresion):
    if max_n_files > 0 and len(all_files) >= max_n_files:
      return all_files
    file_name = str(f) if prepend_folder else f.name
    if substring is None or substring in file_name:
      if include_folders or not os.path.isdir(file_name):
        all_files.append(file_name)
        if progress and len(all_files) % 1000 == 0:
          print("Found {} files".format(len(all_files)))
  return all_files


from vis_utils.simple_3dviz.simple_3dviz.renderables import Spherecloud
from vis_utils.simple_3dviz.simple_3dviz.utils import render
from vis_utils.simple_3dviz.simple_3dviz import Mesh
from vis_utils.simple_3dviz.simple_3dviz.behaviours.io import StoreFramesAsList
from vis_utils.simple_3dviz.simple_3dviz.behaviours.movements import CameraTrajectory
from vis_utils.simple_3dviz.simple_3dviz.behaviours.trajectory import Circle
from vis_utils.simple_3dviz.simple_3dviz import Lines

def render_pointcloud(pcl, colors, K=None, valid_mask=None, add_camera_frustrum=False, up_and_down=False):
  if add_camera_frustrum:
    assert not K is None, "K should be passed when add_camera_frustrum is set to True"
    w, h = K[:2, 2] * 2
    x_0_y_0 = np.linalg.inv(K) @ np.array((0,0,1))
    x_0_y_1 = np.linalg.inv(K) @ np.array((0,h,1))
    x_1_y_0 = np.linalg.inv(K) @ np.array((w,0,1))
    x_1_y_1 = np.linalg.inv(K) @ np.array((w,h,1))

    x_0_y_0 /= np.linalg.norm(x_0_y_0) / 0.3
    x_0_y_1 /= np.linalg.norm(x_0_y_1) / 0.3
    x_1_y_0 /= np.linalg.norm(x_1_y_0) / 0.3
    x_1_y_1 /= np.linalg.norm(x_1_y_1) / 0.3

    l = Lines([
      [0.0, 0.0, 0.0],
      x_0_y_0.tolist(),
      [0.0, 0.0, 0.0],
      x_0_y_1.tolist(),
      [0.0, 0.0, 0.0],
      x_1_y_0.tolist(),
      [0.0, 0.0, 0.0],
      x_1_y_1.tolist(),

      x_0_y_0.tolist(),
      x_1_y_0.tolist(),
      x_1_y_0.tolist(),
      x_1_y_1.tolist(),
      x_1_y_1.tolist(),
      x_0_y_1.tolist(),
      x_0_y_1.tolist(),
      x_0_y_0.tolist(),

    ],
      colors=np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],

        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
      ]), width=0.01)

  assert pcl.shape[0] == colors.shape[0] == 3
  assert pcl.shape[1:] == colors.shape[1:]

  assert valid_mask is None or valid_mask.shape == colors.shape[1:]

  centers = np.array(pcl.reshape(3, -1)).transpose()
  pcl_colors = np.array(colors.reshape(3, -1)).transpose()

  if not valid_mask is None:
    valid_mask = valid_mask.reshape(-1)
    centers = centers[valid_mask == 1, :]
    pcl_colors = pcl_colors[valid_mask == 1, :]

  # scale
  z_at_80 = np.quantile(centers[:, 2], 0.80)
  centers /= z_at_80

  sizes = np.zeros(len(centers)) + 0.002

  s = Spherecloud(centers=centers, sizes=sizes, colors=pcl_colors)

  #m = Mesh.from_file("models/baby_yoda.stl", color=(0.1, 0.8, 0.1))
  #m.to_unit_cube()
  init_camera_pos = (1, 0, 0)
  camera_target = (0, 0, 1)
  objects_to_render = [s]

  if add_camera_frustrum:
    objects_to_render.append(l)

  frame_list_storer = StoreFramesAsList()
  render(objects_to_render,
         n_frames=60,
         camera_position=init_camera_pos,
         camera_target=camera_target,
         up_vector=(0, -1, 0),
         size=(256, 256),
         light=(0,0,0),
         behaviours=[CameraTrajectory(Circle(center=camera_target,
                                             point=init_camera_pos,
                                             normal=[0, -1, 0]), speed=0.004),
                     frame_list_storer])
  frames = [k[:,:,:3].transpose((2,0,1)) for k in frame_list_storer.get_frames()]
  frames = list_of_lists_into_single_list([frames[30:], frames[::-1], frames[0:30]])

  return frames


class ListSampler(torch.utils.data.sampler.Sampler):
  def __init__(self, dataset, n_samples_to_test, debug=False, seed=1337, shuffle_samples=False):
    super(ListSampler, self).__init__(None)
    if n_samples_to_test == 1:
      self.indices = [0]
    else:
      # we do this so that all tests are performed on the same samples, that are randomly selected,
      # when not testing on all samples
      random_state = random.getstate()
      random.seed(seed)
      all_indices = list(range(len(dataset)))
      random.shuffle(all_indices)
      self.indices = all_indices[:n_samples_to_test]
      if shuffle_samples:
        random.shuffle(self.indices)
      elif not debug:
        # if debug do not sort, so that we see more diversity while debugging
        self.indices.sort()
      random.setstate(random_state)


  def __len__(self):
    return len(self.indices)

  def __iter__(self):
    return self.indices.__iter__()


def str2bool(v):
  assert type(v) is str
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')


no_context = contextlib.suppress


def scale_image_biggest_dim(im, biggest_dim):
  #if it is a video, resize inside the video
  if im.shape[1] > im.shape[2]:
    scale = im.shape[1] / (biggest_dim + 0.0)
  else:
    scale = im.shape[2] / (biggest_dim + 0.0)
  target_imshape = (int(im.shape[1]/scale), int(im.shape[2]/scale))
  if im.shape[0] == 1:
    im = myimresize(im[0], target_shape=(target_imshape))[None,:,:]
  else:
    im = myimresize(im, target_shape=target_imshape)
  return im

def myimresize(img, target_shape, interpolation_mode=cv2.INTER_NEAREST):
  max = img.max(); min = img.min()
  uint_mode = img.dtype == 'uint8'

  assert len(target_shape) == 2, "Passed shape {}. Should only be (height, width)".format(target_shape)
  if max > min and not uint_mode:
    # normalize image and undo normalization after the resize
    img = (img - min)/(max - min)
  if len(img.shape) == 3 and img.shape[0] in [1,3]:
    if img.shape[0] == 3:
      img = np.transpose(cv2.resize(np.transpose(img, (1,2,0)), target_shape[::-1], interpolation=interpolation_mode), (2,0,1))
    else:
      img = cv2.resize(img[0], target_shape[::-1], interpolation=interpolation_mode)[None,:,:]
  else:
    img = cv2.resize(img, target_shape[::-1], interpolation=interpolation_mode)
  if max > min and not uint_mode:
    # undo normalization
    return (img*(max - min) + min)
  else:
    return img

def count_trainable_parameters(network, return_as_string=False):
  n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
  if return_as_string:
    return f"{n_parameters:,}"
  else:
    return n_parameters


from torch import data

class CombinedDataset(data.Dataset):
  def __init__(self, datasets):
    assert type(datasets) is list
    self.datasets = datasets

    self.i_to_dataset_and_sample = dict()

    total_i = 0
    for cur_dataset in datasets:
      for k in range(len(cur_dataset)):
        self.i_to_dataset_and_sample[total_i] = (k, cur_dataset)
        total_i += 1

    self.total_samples = total_i

  def __len__(self):
    return self.total_samples

  def __getitem__(self, item):
    i, dataset = self.i_to_dataset_and_sample[item]
    return dataset[i]


def get_conda_env():
  assert 'anaconda' in sys.executable and sys.executable
  return sys.executable.split('/')[-3]