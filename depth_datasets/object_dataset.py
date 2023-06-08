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


import random

import PIL.Image
import cv2

from object_prediction.depth_datasets.abo_renders.abo_renders_dataset import ABORendersDataset
from object_prediction.depth_datasets.abo.abo_dataset import ABODataset
from object_prediction.depth_datasets.hm3_abo.hm3_abo_dataset import HM3AboDataset
from object_prediction.depth_datasets.google_scanned_objects.google_scanned_objects_dataset import GoogleScannedObjectsDataset
from object_prediction.depth_datasets.nerf_sequences.nerf_dataset_sequences import NerfDatasetSequences
from object_prediction.depth_datasets.hndr.hndr_dataset import HNDRDataset
from object_prediction.depth_datasets.image_folder.image_folder_dataset import ImageFolderDataset
from object_prediction.depth_datasets.dtu.dtu import DTUDataset
from object_prediction.depth_datasets.ners.ners_dataset import NersDataset
from object_prediction.depth_datasets.lego_mobile_brick.lego_mobile_dataset import LegoMobileDataset


from object_prediction.single_prediction_from_multi_views.image_configs import *
from object_prediction.depth_metrics_and_losses import normalize_scale_depth

import PIL
from object_prediction.depth_datasets.object_dataset_utils import *


def raise_exception_method():
  raise Exception("To compute number.")

'''
mean_depths_per_dataset = {'abo-renders': 1.4228877451985271,
                           'abo': 1.4265209666163792,
                           'hm3-abo': 1.4785986311446409,
                           'google-scans': 0.24018717458922914,
                           'nerf-sequences': 0.4741111130357545,
                           'hndr': 0.21230325999833063,
                           'dtu': 5.00000001312659
                           'ners': raise_exception_method}
'''

class ObjectDataset():
  def __init__(self, split, resolution=384, dataset_name='abo', img_transform=None, normalize_scale=True, erode_masks=False,
               background_type='original', randomize_configs=False, image_folder=None, debug=False, zero_pad=True, N_textures=200, **kwargs):
    self.resolution = resolution
    self.img_transform = img_transform
    self.normalize_scale = normalize_scale
    self.erode_masks = erode_masks
    self.randomize_configs = randomize_configs

    assert background_type in ['white', 'original', 'random_noise', 'random_texture']

    self.background_type = background_type

    if self.background_type == 'random_texture':
      # load 200 random images from places365
      r_state = random.getstate()
      places_categories = listdir('/data/vision/torralba/datasets/places/files/train', prepend_folder=True, type='folder')
      self.random_images = []
      print("Using random texture. Will list {} textures from places".format(N_textures))
      for c in tqdm(random.sample(places_categories, N_textures)):
        img_path = random.choice(listdir(c, prepend_folder=True, extension='.jpg'))
        img = best_centercrop_image(cv2_imread(img_path), resolution, resolution)
        self.random_images.append(img)
      random.setstate(r_state)

    self.debug = debug

    self.dataset_name = dataset_name
    self.zero_pad = zero_pad

    assert dataset_name in ['abo-renders', 'abo', 'hm3-abo', 'google-scans', 'nerf-sequences', 'image-folder', 'hndr', 'dtu', 'ners', 'lego-mobile'], "Dataset {} not available!".format(dataset_name)
    if dataset_name == 'abo':
      self.dataset = ABODataset(split=split, **kwargs)
    elif dataset_name == 'hm3-abo':
      self.dataset = HM3AboDataset(split=split, **kwargs)
    elif dataset_name == 'google-scans':
      self.dataset = GoogleScannedObjectsDataset(split=split, **kwargs)
    elif dataset_name == 'nerf-sequences':
      self.dataset = NerfDatasetSequences(split=split, **kwargs)
    elif dataset_name == 'hndr':
      self.dataset = HNDRDataset(split=split, **kwargs)
    elif dataset_name == 'dtu':
      self.dataset = DTUDataset(split=split, **kwargs)
    elif dataset_name == 'abo-renders':
      self.dataset = ABORendersDataset(split=split)
    elif dataset_name == 'ners':
      self.dataset = NersDataset(split=split)
    elif dataset_name == 'lego-mobile':
      self.dataset = LegoMobileDataset(split=split)
    elif dataset_name == 'image-folder':
      assert not image_folder is None and os.path.isdir(image_folder), "Image folder {} is not a directory!".format(image_folder)
      self.dataset = ImageFolderDataset(image_folder)

    self.random_config_generator = None

    # cache so that
    self.last_return_item_id = -1
    self.last_base_item = None

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, item):
    if item == self.last_return_item_id:
      # we cache previous call, so that we can generate randomized configs for the same item more efficiently
      base_item = self.last_base_item
    else:
      base_item = self.dataset[item]
      self.last_return_item_id = item
      self.last_base_item = base_item

    if self.dataset_name == 'dtu':
      # for dtu, the depth masks and object masks are different, as not depth is valid for the gt.
      img, depth, object_mask, depth_mask, normals, K, datset_info = base_item
    else:
      img, depth, depth_mask, normals, K, datset_info = base_item

    assert img.max() <= 255.0, "Error on dataset returning image"

    assert img.shape[-2] == img.shape[-1], "The processing assumes that the dataset returns squared images"

    perspective_transform = None
    if self.randomize_configs:
      if self.random_config_generator is None:
        self.random_config_generator = RandomConfigGenerator(img.shape[1:], K,
                                                             debug=False,
                                                             rotation_prob=0.8,
                                                             zoom_prob=0.8,
                                                             max_zoom_factor=4,
                                                             illum_prob=0)
      try:
        assert self.dataset == 'dtu', "Needs to implement for dtu, by also adding object_mask to the transform"
        random_config = self.random_config_generator.generate_random_config(img, object_mask)
        img, object_mask, depth_mask, depth, K, normals = random_config.apply_config(img, object_mask, depth_mask, depth, K, normals)
        perspective_transform = random_config.get_perspective_transform(img.shape[1:])
      except Exception as e:
        # if it fails, just just default image
        print("Failed to generate random config, will use default image")
        print("Image file: {}".format(datset_info['img_file']))
        print("Exception")
        print(e)

    if perspective_transform is None:
      perspective_transform = ImageConfig.identity_perspective_transform()
      random_config = EmptyConfig()

    if self.dataset_name != 'dtu':
      object_mask = np.array(depth_mask)

    if self.background_type == 'white':
      img[:, np.array(1 - object_mask, dtype='bool')] = 255
    elif self.background_type == 'random_noise':
      shape_to_replace = img[:, np.array(1 - object_mask, dtype='bool')].shape
      pixel_values = np.array(np.random.uniform(0,1,size=shape_to_replace) * 255, dtype='uint8')
      img[:, np.array(1 - object_mask, dtype='bool')] = pixel_values
    if self.background_type == 'random_texture':
      mask = np.array(1 - object_mask, dtype='bool')
      background_img = cv2_resize(random.choice(self.random_images), img.shape[1:])
      img[:, mask] = background_img[:, mask]

    resize_ratio = self.resolution / img.shape[1]
    K[:2, :] *= resize_ratio

    if self.debug:
      imshow(img * depth_mask, "masked_before_resizing")

    interpolation_mode = cv2.INTER_NEAREST # cv2.INTER_LINEAR_EXACT

    assert img.shape[1:] == depth_mask.shape

    img = myimresize(img, (self.resolution, self.resolution), interpolation_mode=interpolation_mode)
    depth_mask = myimresize(depth_mask * 1.0, (self.resolution, self.resolution), interpolation_mode=interpolation_mode)
    object_mask = myimresize(object_mask * 1.0, (self.resolution, self.resolution), interpolation_mode=interpolation_mode)

    if self.debug:
      imshow(img * depth_mask, "masked_after_resizing")

    if not depth is None:
      depth = myimresize(depth, (self.resolution, self.resolution), interpolation_mode=interpolation_mode)
      normals = myimresize(normals, (self.resolution, self.resolution), interpolation_mode=interpolation_mode)

      if self.normalize_scale and depth_mask.sum() > 0:
        depth = normalize_scale_depth(depth, depth_mask)

    img_to_transform = tonumpy(img).transpose((1, 2, 0))
    # pil_img = Image.fromarray(img_to_transform)
    if not self.img_transform is None:
      try:
        # for transforms that expect a dict with image, depth,...
        # as is the case of dpt transform
        transformed_img = self.img_transform({'image': img_to_transform / 255.0})['image']
      except:
        # for other transforms that directly expect an image
        assert img_to_transform.dtype == 'uint8' and img_to_transform.shape[-1] == 3
        pil_image = PIL.Image.fromarray(img_to_transform)
        transformed_img = self.img_transform(pil_image)
    else:
      transformed_img = img_to_transform.transpose((2, 0, 1))

    if self.erode_masks:
      k_s = int(10 * self.resolution / 384)
      k = np.ones((k_s, k_s), np.uint8)
      original_depth_mask = depth_mask
      depth_mask = cv2.erode(depth_mask, k)

    assert transformed_img.shape == img.shape, "The transform should keep the image shape as it was before the transform," \
                                               "or depth/masks transformation needs implementing!"

    R = np.eye(3)
    zoom_factor = 1
    if not random_config is None and type(random_config) is ComposedConfig:
      for c in random_config.config_list:
        if type(c) is ZoomConfig:
          zoom_factor = c.zoom_factor
        elif type(c) is PureRotationConfig:
          R = c.get_rotation()


    if not depth is None:
      depth = torch.FloatTensor(depth)
    else:
      depth = torch.zeros(1)
    if not normals is None:
      normals = torch.FloatTensor(normals)
    else:
      normals = torch.zeros(1)


    return torch.FloatTensor(transformed_img), \
           torch.FloatTensor(object_mask), \
           depth, \
           torch.FloatTensor(depth_mask), \
           normals, \
           torch.FloatTensor(K), \
           torch.FloatTensor(perspective_transform), \
           torch.FloatTensor(R), \
           torch.FloatTensor([zoom_factor])


def compute_average_depth_normalization(n_samples=1000, normalize_scale=True, dataset_to_test=None, parallel=True):
  if dataset_to_test is None:
    datasets_to_test = ['abo-renders', 'abo', 'hm3-abo', 'google-scans', 'nerf-sequences', 'hndr', 'dtu']
  elif type(dataset_to_test) is list:
    datasets_to_test = dataset_to_test
  else:
    datasets_to_test = [dataset_to_test]
  for dataset_name in datasets_to_test:
    dataset = ObjectDataset(dataset_name=dataset_name,
                            split='val',
                            randomize_configs=False,
                            background_type='white',
                            debug=False,
                            normalize_scale=normalize_scale)

    samples = list(range(len(dataset)))
    if n_samples < len(samples):
      samples = random.sample(samples, n_samples)

    def get_one_depth(item):
      _, _, depth, depth_mask, normals, K, transform, R, zoom_factor = dataset[item]
      return tonumpy(depth[depth_mask == 1.0]).tolist()

    all_depths = process_in_parallel_or_not(get_one_depth, samples, parallel=parallel)
    all_means = [np.mean(d) for d in all_depths]
    all_means = [k for k in all_means if not np.isnan(k)]
    overall_mean = np.mean(all_means)

    print("Mean depth {}{}: {}".format('normalized (should be close to 1) ' if normalize_scale else '', dataset_name, overall_mean))
    # visdom_histogram(all_depths, title=dataset_name + 'depth_histogram')


if __name__ == '__main__':
  # compute_average_depth_normalization(n_samples=1000, normalize_scale=True, dataset_to_test=['abo'])
  background_type = 'random_texture'

  datasets_to_test = ['abo-renders', 'abo', 'hm3-abo', 'google-scans', 'nerf-sequences', 'dtu', 'ners', 'lego-mobile']
  for dataset_name in ['ners']:
    from torchvision import transforms
    dataset = ObjectDataset(dataset_name=dataset_name,
                            split='val',
                            randomize_configs=False,
                            background_type=background_type,
                            debug=True,
                            N_textures=10) #,
                            #img_transform=transforms.Compose(transforms.ToTensor()))

    elements = list(range(len(dataset)))
    random.shuffle(elements)
    for item_i in elements:
      img, object_mask, gt_depth, depth_mask, gt_normal, K, perspective_transform, _, _ = dataset[item_i]
      imshow(img, title='img')
      imshow(img * object_mask[None,...], title='img_masked')

      # continue

      imshow(object_mask, title='object_mask')
      imshow(gt_depth, title='gt_depth')
      imshow(depth_mask, title='depth_mask')
      imshow(gt_normal, title='gt_normal')

      assert img.shape == (3,384,384)
      assert object_mask.shape == (384,384)
      assert gt_depth.shape == (384,384)
      assert depth_mask.shape == (384,384)
      assert gt_normal.shape == (3,384,384)

      pcl = pixel2cam(totorch(gt_depth)[None, ...], totorch(K[None]))[0]
      show_pointcloud(pcl, img, valid_mask=depth_mask)
