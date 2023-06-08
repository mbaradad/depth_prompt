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


import sys
sys.path.append('.')

from utils import *

import torch.utils.data
import gzip
import json

import random

from object_prediction.depth_datasets.object_dataset_utils import *

import pandas as pd

ABO_BASE_PATH='datasets/amazon_berkeley_objects'
ABO_CACHE='datasets/amazon_berkeley_objects_cache'

# not all objects have 3 scenes, see ABO_BASE_PATH/train_test_split.csv
# N_SCENES = 3
N_VIEWS_PER_SCENE = 91

class ABODataset:
  def __init__(self, split, ids=None):
    assert split in ['train', 'val']
    assert ids is None or type(ids) is list, "Ids should be None or a list of valid object id's"
    train_test_objects = read_text_file_lines(ABO_BASE_PATH + '/train_test_split.csv')


    train_objects_and_scenes = []
    val_objects_and_scenes = []
    for l in train_test_objects[1:]:
      object_name, obj_split, N_scenes, scene_idxs = l.split(',')
      if not ids is None and not object_name in ids:
        continue
      scene_idxs = scene_idxs.split('_')

      for idx in scene_idxs:
        if obj_split == 'TRAIN':
          train_objects_and_scenes.append((object_name, idx))
        elif obj_split == 'TEST':
          val_objects_and_scenes.append((object_name, idx))
        else:
          raise Exception("Split {} invalid".format(obj_split))

    if split == 'train':
      self.objects_and_scenes = train_objects_and_scenes
    else:
      self.objects_and_scenes = val_objects_and_scenes

    self.objects_and_scenes.sort()

    print("Loading metadata for all {} objects in split {}".format(self. get_n_objects(), split))
    metadata_cache = ABO_CACHE + '/metadata_cache.pckl'
    if os.path.exists(metadata_cache):
      self.metadata = load_from_pickle(metadata_cache)
    else:
      os.makedirs(ABO_CACHE, exist_ok=True)
      self.metadata = dict()
      for object, _ in self.objects_and_scenes:
        # K is the same for all, so no need to load all metadata for all objects
        self.metadata[object] = load_json('{}/all_objects/{}/metadata.json'.format(ABO_BASE_PATH, object))
        self.K = np.array(self.metadata[object]['views'][0]['K']).reshape(3,3)
        # check self.K is invertible, if not something is wrong:
        # https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
        assert np.linalg.cond(self.K) < 1/sys.float_info.epsilon, "K matrix is not invertible (K = {})".format(self.K)

        break
      # dump_to_pickle(metadata_cache, self.metadata)

    self.object_id_to_object_name = None

  @staticmethod
  def get_intrinsics(h, w):
    assert h == w, "Abo is squared. Needs reimplementation/testing for non-squared functionality"
    original_w = 512
    K = np.array([[443.40496826, 0., 256.],
                  [0., 443.40496826, 256.],
                  [0., 0., 1.]])

    resize_ratio = w / original_w

    K *= resize_ratio
    K[2,2] = 1

    return K

  def _load_amazon_metadata(self):
    print("Loading amazon metadata")
    amazon_metadata_files = listdir(ABO_BASE_PATH + '/listings/metadata', prepend_folder=True)
    model_id_tag = '3dmodel_id'
    df = None
    for m_file in tqdm(amazon_metadata_files):
      cur_df = pd.read_json(
        m_file,
        lines=True,
        compression='gzip'
      )
      cur_df = cur_df.dropna(subset=[model_id_tag])
      if df is None:
        df = cur_df
      else:
        df = pd.concat([df, cur_df])
    df.set_index(model_id_tag)

    self.object_id_to_object_name = dict()
    for object_id, row in df.iterrows():
      object_name = 'Unknown'
      for name_by_language in row['item_name']:
        if name_by_language['language_tag'] in ['en_US', 'en_GB']:
          object_name = name_by_language['value']
          break
      self.object_id_to_object_name[row[model_id_tag]] = object_name

  def __len__(self):
    return len(self.objects_and_scenes) * N_VIEWS_PER_SCENE

  def get_n_objects(self):
    return len(set([k[0] for k in self.objects_and_scenes]))

  def get_object_amazon_metadata(self, object_id):
    if self.object_id_to_object_name is None:
      self._load_amazon_metadata()
    return self.object_id_to_object_name[object_id]

  def __getitem__(self, item):
    object_and_scene_n = item // N_VIEWS_PER_SCENE
    render_id = item % N_VIEWS_PER_SCENE

    object_id, scene_n = self.objects_and_scenes[object_and_scene_n]

    rendered_img_f = '{}/all_objects/{}/render/{}/render_{}.jpg'.format(ABO_BASE_PATH, object_id, scene_n, render_id)
    depth_f = '{}/all_objects/{}/depth/depth_{}.exr'.format(ABO_BASE_PATH, object_id, render_id)
    normal_f = '{}/all_objects/{}/normal/normal_{}.png'.format(ABO_BASE_PATH, object_id, render_id)

    img = cv2_imread(rendered_img_f)

    normals = cv2_imread(normal_f) / (255.0 / 2 ) - 1
    normals[0] = -1 * normals[0]

    depth = open_exr_depth(depth_f)

    depth_mask = depth < 65000

    depth[np.logical_not(depth_mask)] = 0
    normals[:, np.logical_not(depth_mask)] = 0

    assert img.shape[1] == img.shape[2], "Only implemented for squared images"

    # K is the same for all examples
    # cur_meta = self.metadata[object_id]['views'][render_id]
    K = np.array(self.K)

    depth_mask = depth_mask * 1.0

    dataset_info = {'object_id': object_id,
                   'render_id': render_id,
                   'img_file': rendered_img_f}

    return img, \
           depth, \
           depth_mask, \
           normals, \
           K, \
           dataset_info

def load_all_data_and_get_stats(split='train', batch_size=16, num_workers=16):
  dataset = ABODataset(split=split)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

  min_depths, max_depths, mean_depths, depth_ranges, fov_xs = [], [], [], [], []
  for i, (imgs, object_masks, depths, depth_masks, normals, Ks, object_ids, render_ids) in enumerate(tqdm(loader)):
    for img, d, d_mask, K, object_id, render_id in zip(imgs, depths, depth_masks, Ks, object_ids, render_ids):
      valid_depths = d[d_mask.bool()]
      if len(valid_depths) == 0:
        continue

      fov_x_deg = intrinsics_to_fov_x_deg(K)

      min_depth = valid_depths.min()
      max_depth = valid_depths.max()
      depth_range = max_depth - min_depth


      min_depths.append(min_depth)
      max_depths.append(max_depth)
      mean_depths.append(valid_depths.mean())
      depth_ranges.append(depth_range)
      fov_xs.append(fov_x_deg)


      #if depth_range < 0.01:
      #  print("Small object with id: {}, render_{}, description {}".format(object_id, int(render_id), dataset.get_object_amazon_metadata(object_id)))
      #  imshow(img)

    if (i + 1) % 100 == 0:
      visdom_histogram(min_depths, title='min_depths')
      visdom_histogram(max_depths, title='max_depths')
      visdom_histogram(mean_depths, title='mean_depths')
      visdom_histogram(depth_ranges, title='depth_ranges')
      visdom_histogram(fov_xs, title='fov_xs')


if __name__ == '__main__':
  #test_all_and_get_stats('val')
  #test_all_and_get_stats('train')
  #exit()

  from dpt.dpt.transforms import Resize, NormalizeImage, PrepareForNet
  from torchvision.transforms import Compose

  normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  net_w = net_h = 384
  img_transform = Compose(
    [
      Resize(
        net_w,
        net_h,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="minimal",
        image_interpolation_method=cv2.INTER_CUBIC,
      ),
      normalization,
      PrepareForNet(),
    ]
  )


  dataset = ABODataset(split='train')
  indices = list(range(len(dataset)))

  random.shuffle(indices)

  n_valid_pixels_ratio = []

  batch_size = 32
  test_loader_speed = False
  if test_loader_speed:
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=32, pin_memory=True, drop_last=True)

    t_0 = time.time()
    for i, b in enumerate(train_loader):
      t_1 = time.time()
      run_time = t_1 - t_0
      print("{} elems/s".format(batch_size * (i + 1) / run_time))
  else:
    for i in indices:
      img, object_mask, depth, depth_mask, normals, K = dataset.__getitem__(i)
      imshow(img * object_mask, title='masked_object', biggest_dim=600)
      continue

      if depth_mask.sum() == 0:
        continue
      n_valid_pixels_ratio.append(depth_mask.mean())

      print("Showing sample "+ str(i))

      pcl = pixel2cam(totorch(depth)[None, ...], totorch(K[None]))[0]
      normals_from_pcl = compute_normals_from_closest_image_coords(pcl[None])[0]

      normalized_img = (img - img.min()) / (img.max() - img.min())

      show_pointcloud(pcl, normalized_img, valid_mask=depth_mask, title='pcl')

      imshow(depth_mask, title='mask')
      imshow(img, title='image')
      normals[0] = normals[0]
      imshow(normals, title='normals')
      imshow(normals_from_pcl, title='normals_from_pcl')

      imshow(depth, title='depth')

      cos_sim_between_normals = ((normals[:, :-1,:-1] * normals_from_pcl).sum(0) * depth_mask[:-1,:-1]).sum() / (depth_mask.sum() + 1e-6)

      print("Cos sim beteween gt normals and normals from pcl (best is 1): {}".format(cos_sim_between_normals))
      if cos_sim_between_normals < 0.8:
        a = 1