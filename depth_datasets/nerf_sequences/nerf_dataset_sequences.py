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

from object_prediction.depth_datasets.object_dataset_utils import *

NERFIES_DUMPED_PATH='datasets/nerfies/nerfies_depths'
NERFIES_SEGMENTATIONS='datasets/nerfies/nerfies_foreground_segmentations'

class NerfDatasetSequences():
  def __init__(self, split, min_foreground_percentage=0.05, debug=False):
    assert split == 'val', "Nerf data is only for testing, as there are not that many sequences!"
    self.debug = debug

    self.scenes = listdir(NERFIES_DUMPED_PATH, prepend_folder=True, type='folder')
    self.all_examples = list_of_lists_into_single_list([listdir(scene, prepend_folder=True) for scene in self.scenes])

    self.min_foreground_percentage = min_foreground_percentage

    self._make_all_segmentations_and_filter_examples()

  @staticmethod
  def get_foreground_file_from_nerfie_file(file):
    return file.replace('/nerfies_depth', '/nerfies_foreground_segmentations')

  def _make_all_segmentations_and_filter_examples(self):
    for example in tqdm(self.all_examples):
      foreground_file = self.get_foreground_file_from_nerfie_file(example)
      if not os.path.exists(foreground_file):
        print("Some segmentation mask not found, will compute them! This may not free the memory of the model after "
              "usage, so it is recommended that you run the main before using the class!")
        img = np.load(example)['rgb']
        from remove_background.bg_remover import remove_background

        segmented_image, foreground = remove_background(np.array(img * 255.0, dtype='uint8'), binary_mask=True)

        make_dir_without_file(foreground_file)

        np.savez_compressed(foreground_file, foreground=foreground)

    is_valid_example_file = '{}/examples_list_with_foreground_percentage'.format(NERFIES_DUMPED_PATH)
    valid_examples_found = False
    if os.path.exists(is_valid_example_file):
      examples_with_foreground_percentage = read_text_file_lines(is_valid_example_file)
      examples_with_foreground_percentage = [k.replace('/home/mbj_google_com/projects/normals_acc/',
                                                       '/data/vision/torralba/movies_sfm/home/normals_acc/') for k in examples_with_foreground_percentage]

      if len(examples_with_foreground_percentage) == len(self.all_examples):
        valid_examples_found = True
        examples_with_foreground_percentage = dict([(l[0], float(l[1])) for l in [tuple(k.split(',')) for k in examples_with_foreground_percentage]])

    if not valid_examples_found:
      examples_with_foreground_percentage = dict()
      for example in tqdm(self.all_examples):
        foreground_file = self.get_foreground_file_from_nerfie_file(example)
        foreground = np.load(foreground_file)['foreground']
        examples_with_foreground_percentage[example] = foreground.mean()
      write_text_file_lines(['{},{}'.format(example, examples_with_foreground_percentage[example]) for example in self.all_examples], is_valid_example_file)

    final_examples = []
    for example, foreground_percentage in examples_with_foreground_percentage.items():
      if foreground_percentage > self.min_foreground_percentage:
        final_examples.append(example)

    self.examples = final_examples

  def __len__(self):
    return len(self.examples)


  def __getitem__(self, item):
    example_file = self.examples[item]
    example = np.load(example_file)
    img, depth, K = [example[k] for k in ['rgb', 'depth', 'K']]
    foreground = np.load(self.get_foreground_file_from_nerfie_file(example_file))['foreground']

    img = crop_to_square(img)
    depth = crop_to_square(depth[None])[0]
    foreground = crop_to_square(foreground[None])[0]

    depth_mask = foreground

    if K[0,2] > K[1,2]:
      K[0, 2] = K[1, 2]
    else:
      K[1, 2] = K[0, 2]

    normals_f = NERFIES_DUMPED_PATH.replace('_depths', '_normals') + '/' + '/'.join(example_file.split('/')[-2:])
    normals_folder = '/'.join(normals_f.split('/')[:-1])
    os.makedirs(normals_folder, exist_ok=True)

    if os.path.exists(normals_f):
      normals = np.load(normals_f)['normals']
    else:
      pcl = pixel2cam(totorch(depth)[None,:,:], totorch(K)[None,:,:])
      computed_normals = compute_normals_from_closest_image_coords(pcl)[0]
      normals_padded = np.zeros_like(img, dtype='float32')
      normals_padded[:,1:,1:] = computed_normals
      os.makedirs('/'.join(normals_f.split('/')[:-1]), exist_ok=True)
      if not self.debug:
        np.savez_compressed(normals_f, normals=normals_padded)
      else:
        show_pointcloud(pcl[0], img, title='pcl_from_depth')

      normals = normals_padded

    dataset_info = {'img_file': example_file}

    return np.array(img * 255.0, dtype='uint8'), \
         depth, \
         depth_mask, \
         normals, \
         K, \
         dataset_info

if __name__ == '__main__':
  dataset = NerfDatasetSequences('val', debug=False)
  dataset_elems = list(range(len(dataset)))
  random.shuffle(dataset_elems)
  for item_i in tqdm(dataset_elems):
    img, depth, depth_mask, normals, K, dataset_info = dataset[item_i]
    imshow(img, title='original_img')
    imshow(img * depth_mask, title='masked_img')

    pcl = pixel2cam(totorch(depth[None, ...]), totorch(K[None, ...]))[0]

    show_pointcloud(pcl, img, title='pcl')