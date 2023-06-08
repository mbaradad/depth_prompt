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


from object_prediction.depth_datasets.abo.abo_dataset import *
from collections import defaultdict

class AboSingleClassDataset():
  def __init__(self, split='train', class_name='table'):
    assert split in ['train', 'val']

    file_to_class = [k.split(',') for k in read_text_file_lines(ABO_BASE_PATH + '/abo_classes_3d.txt')]
    class_ids = []
    for file, obj_class in file_to_class:
      if obj_class == class_name:
        class_ids.append(file)

    assert len(class_ids) > 0, "No object found for class {}".format(class_name)

    self.dataset_filtered_by_class = ABODataset(split, class_ids)
    if len(self.dataset_filtered_by_class) == 0:
      print("No objects found for class {} in split {}".format(class_name, split))

  def __len__(self):
    return len(self.dataset_filtered_by_class)

  def __getitem__(self, item):
    self.dataset_filtered_by_class[item]

  @staticmethod
  def get_all_available_classes(return_counts=True):
    file_to_class = read_text_file_lines(ABO_BASE_PATH + '/abo_classes_3d.txt')
    classes = [k.split(',') for k in file_to_class]
    elements_per_class = defaultdict(int)
    for c in classes:
      elements_per_class[c[1]] += 1
    class_n_elements = [(k[1], k[0]) for k in list(elements_per_class.items())]
    class_n_elements.sort()
    class_n_elements = class_n_elements[::-1]

    if return_counts:
      return [(k[1], k[0]) for k in class_n_elements]
    else:
      return [k[1] for k in class_n_elements]

if __name__ == '__main__':
  a = 1
  available_classes_with_counts = AboSingleClassDataset.get_all_available_classes()
  top_classes = available_classes_with_counts[:10]
  for cur_class in top_classes:
    single_class_dataset = AboSingleClassDataset('train', cur_class[0])
