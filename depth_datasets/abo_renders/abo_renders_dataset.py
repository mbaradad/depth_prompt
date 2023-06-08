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

from depth_datasets.abo.abo_dataset import *
from depth_datasets.hm3_abo.hm3_abo_dataset import *

from dpt.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

class ABORendersDataset:
  # mix of abo and abo_renders
  def __init__(self, split, balance_train=True):
    assert split in ['train', 'val']

    # ABO dataset contains ~10 times more data, if balance_train is set to True,
    # use as 10 times less of
    self.hm3_abo_dataset = HM3AboDataset(split)
    self.abo_dataset = ABODataset(split)

    self.balance_train = balance_train

    if balance_train:
      self.abo_dataset_n_per_split = len(self.abo_dataset) // 10

  def __len__(self):
    if self.balance_train:
      return len(self.hm3_abo_dataset) + self.abo_dataset_n_per_split
    else:
      return len(self.hm3_abo_dataset) + len(self.abo_dataset)


  def __getitem__(self, item):
    if item < len(self.hm3_abo_dataset):
      return self.hm3_abo_dataset.__getitem__(item)
    else:
      if self.balance_train:
        subset = np.random.randint(0,10)
        abo_item = subset * self.abo_dataset_n_per_split + item - len(self.hm3_abo_dataset)

      else:
        abo_item = item - len(self.hm3_abo_dataset)

      return self.abo_dataset.__getitem__(abo_item)


if __name__ == '__main__':
  dataset = ABORendersDataset(split='train', balance_train=True)

  normalization = NormalizeImage(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
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
      img, depth, depth_mask, normals, K, datset_info = dataset.__getitem__(i)
      imshow(img * depth_mask, title='masked_object', biggest_dim=600)
      continue

      if depth_mask.sum() == 0:
        continue
      n_valid_pixels_ratio.append(depth_mask.mean())

      print("Showing sample " + str(i))

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