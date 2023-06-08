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
from utils import *

import random
import OpenEXR as exr

HM3ABO_PATH = 'datasets/hm3d_abo'
HM3ABO_DUMP_PATH = 'datasets/dumped_datasets/hm3d_abo'

from object_prediction.depth_datasets.object_dataset_utils import *

class HM3AboDataset():
  def __init__(self, split, debug=False):
    assert split in ['train', 'val']
    self.debug = debug
    train_scenes = read_text_file_lines(HM3ABO_PATH + '/index/train.txt')

    all_scenes = listdir(HM3ABO_PATH + '/scenes', prepend_folder=True)
    train_scenes = [k for k in all_scenes if k.split('/')[-1] in train_scenes]

    if split == 'train':
      self.scenes = train_scenes
    elif split == 'val':
      self.scenes = [k for k in all_scenes if not k in train_scenes]

    all_split_examples_file = HM3ABO_PATH +'/all_examples_' + split
    if os.path.exists(all_split_examples_file):
      self.examples = read_text_file_lines(all_split_examples_file)
    else:
      print("Listing all scenes for HM3 Abo dataset for split {}".format(split))
      self.examples = []
      for s in tqdm(self.scenes):
        self.examples.extend(listdir(s + '/rgb', prepend_folder=True))
      write_text_file_lines(self.examples, all_split_examples_file)

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, item):
    rendered_f = self.examples[item]

    img = cv2_imread(rendered_f)
    depth = open_exr_depth(rendered_f.replace('/rgb/', '/depth/').replace('.jpg', '.exr'))
    invalid_depth = depth > 65000
    depth[invalid_depth] = 0

    depth_mask = cv2_imread(rendered_f.replace('/rgb/', '/mask/').replace('.jpg', '.png'))[0] == 255
    K = np.loadtxt('/'.join(rendered_f.split('/')[:-2]) + '/intrinsic.txt')

    normals_f = HM3ABO_DUMP_PATH + '/' + '/'.join(rendered_f.split('/')[-3:]).replace('/rgb/', '/normals/').replace('.jpg', '.npz')
    normals_folder = '/'.join(normals_f.split('/')[:-1])
    os.makedirs(normals_folder, exist_ok=True)
    normals = None
    if os.path.exists(normals_f):
      try:
        normals = np.load(normals_f)['normals']
      except:
        pass
    if normals is None:
      pcl = pixel2cam(totorch(depth)[None,:,:], totorch(K)[None,:,:])
      computed_normals = compute_normals_from_closest_image_coords(pcl)[0]
      normals_padded = np.zeros_like(img, dtype='float32')
      normals_padded[:,1:,1:] = computed_normals
      os.makedirs('/'.join(normals_f.split('/')[:-1]), exist_ok=True)

      np.savez_compressed(normals_f, normals=normals_padded)

      normals = normals_padded

    dataset_info = {'img_file': rendered_f}

    # center crop to make it compatible with other datasets:
    assert img.shape == (3,480,640)
    img = img[:,:,80:-80]
    depth = depth[:, 80:-80]
    depth_mask = depth_mask[:, 80:-80]
    normals = normals[:, :, 80:-80]
    K[0,2] -= 80

    return img, \
         depth, \
         depth_mask, \
         normals, \
         K, \
         dataset_info

if __name__ == '__main__':
  dataset = HM3AboDataset('train', debug=False)
  dataset_elems = list(range(len(dataset)))
  random.shuffle(dataset_elems)
  for item_i in tqdm(dataset_elems):
    img, depth, depth_mask, normals, K, dataset_info = dataset[item_i]