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

RENDERED_OBJECTS_PATH = 'datasets/dumped_datasets/google_scans_dumped'
KATAMARI_OBJECTS_FOLDER = 'datasets/scanned_objects'


def get_all_objects(prepend_folder=False):
  all_objects = listdir(KATAMARI_OBJECTS_FOLDER, prepend_folder=prepend_folder)
  random.shuffle(all_objects)

  return all_objects

def get_all_rendered_examples():
  return listdir(RENDERED_OBJECTS_PATH, prepend_folder=True)

if __name__ == '__main__':
  rendered_samples = get_all_rendered_examples()


def get_all_rendered_examples():
  return listdir(RENDERED_OBJECTS_PATH, prepend_folder=True)

class GoogleScannedObjectsDataset():
  def __init__(self, split):
    assert split in ['train', 'val']
    r_state = random.getstate()
    random.seed(1337)
    all_examples = get_all_rendered_examples()
    random.shuffle(all_examples)
    if split == 'train':
      self.examples = [k for i, k in enumerate(all_examples) if i % 10 != 0]
    elif split == 'val':
      self.examples = [k for i, k in enumerate(all_examples) if i % 10 == 0]
    random.setstate(r_state)

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, item):
    rendered_f = self.examples[item]
    example = np.load(rendered_f)

    img = example['img']
    depth = example['depth']
    depth_mask = example['depth_mask']
    K = example['K']

    normals_f = rendered_f.replace(rendered_f.split('/')[-2], rendered_f.split('/')[-2] + '_normals')

    if os.path.exists(normals_f):
      normals = np.load(normals_f)['normals']
    else:
      pcl = pixel2cam(totorch(depth)[None,:,:], totorch(K)[None,:,:])
      computed_normals = compute_normals_from_closest_image_coords(pcl)
      normals_padded = np.zeros_like(img, dtype='float32')
      normals_padded[:,1:,1:] = computed_normals
      os.makedirs('/'.join(normals_f.split('/')[:-1]), exist_ok=True)
      np.savez_compressed(normals_f, normals=normals_padded)

      normals = normals_padded

    dataset_info = {'img_file': rendered_f}

    assert normals.shape == img.shape, "Normals are not padded to have the same shape as the image"
    assert img.max() <= 1.0
    img = np.array(img * 255.0, dtype='uint8')

    return img , \
           depth, \
           depth_mask, \
           normals, \
           K, \
           dataset_info

if __name__ == '__main__':
  dataset = GoogleScannedObjectsDataset('train')
  for i, (img, depth, depth_mask, normals, K, dataset_info) in enumerate(tqdm(dataset)):
    imshow(img)
