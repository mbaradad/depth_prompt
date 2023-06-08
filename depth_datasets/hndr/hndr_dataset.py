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
from utils import *
sys.path.append('./datasets/hndr/hndr_code')

HNDR_PATH = 'datasets/hndr_depth_data'
HNDR_CACHE_PATH = 'datasets/hndr_depth_data_computed'

from object_prediction.depth_datasets.object_dataset_utils import *
from remove_background.rembg_bg_remover import remove_background_rembg

class HNDRDataset():
  def __init__(self, split='val', debug=False, zero_pad=True):
    assert split in ['val'], 'HNDR should only be used for evalutaion!'
    self.debug = debug
    self.scenes = listdir(HNDR_PATH, prepend_folder=False, type='folder')

    assert len(self.scenes) == 10, "There are only 10 scenes in the dataset (found {}). Some missing or extra folder is there!".format(self.scenes)
    self.debug = debug
    self.zero_pad = zero_pad

  def __len__(self):
    return len(self.scenes)

  def get_computed_depth_and_image(self, scene_name, device="cuda"):
    cache_file = HNDR_CACHE_PATH + '/' + scene_name + ('_zero_pad' if self.zero_pad else '') +'.npz'

    loaded = False
    if os.path.exists(cache_file) and not self.debug:
      try:
        scene_things = np.load(cache_file)
        loaded = True
      except: pass
    if not loaded:
      print("Cache file not found for scene {}, will compute it which uses GPU in the training data".format(scene_name))
      bundle = np.load("{0}/{1}/frame_bundle.npz".format(HNDR_PATH, scene_name), allow_pickle=True)

      original_img = bundle['img_0'].transpose((2,0,1))
      original_depth = bundle['depth_0']
      conf = bundle['conf_0']
      info = dict(bundle['info_0'].tolist())

      timestamp = info['timestamp']
      euler_angles = info['euler_angles']
      world_pose = info['world_pose']
      K = info['intrinsics']
      world_to_camera = info['world_to_camera']

      reprojected_lidar_depth = np.load("{0}/{1}/reprojected_lidar.npy".format(HNDR_PATH, scene_name))  # point to reprojected lidar depth

      model = torch.load("{0}/{1}/model.pt".format(HNDR_PATH, scene_name), map_location="cuda")  # point to location of trained model
      model.args.device = device  # perform reconstruction on cpu
      model.ref_depth = torch.tensor(reprojected_lidar_depth).float()[None, None].to(device)  # use reprojected lidar depth as intialization
      _, original_height, original_width = original_img.shape

      # max_width = original_width // 2
      max_width = 720
      resize_ratio = (max_width / original_width)
      resized_height = int(original_height * resize_ratio)

      K = K * resize_ratio
      K[2,2] = 1

      img = cv2_resize(original_img,
                       target_height_width=(int(np.ceil(resized_height)), max_width))

      qry, out = model.get_visualization(y_samples=resized_height,
                                              x_samples=max_width)  # set resolution, full resolution might require large memory
      out = out.detach()
      qry = qry.detach()

      reprojected_lidar_depth = cv2_resize(tonumpy(reprojected_lidar_depth), (resized_height, max_width))

      segmented_image, foreground = remove_background_rembg(img, binary_mask=True)
      foreground_percentage = foreground.mean()
      assert foreground_percentage > 0.05, "Img {} only has {} foreground percentage, minimum required: {}".format(
                                                                                original_img,
                                                                                foreground_percentage,
                                                                                0.05)


      _, ori_h, ori_w = img.shape
      if ori_h != ori_w:
        if self.zero_pad:
          assert foreground.shape == img.shape[1:], "Img and foreground shape must match before zero padding"
          if ori_h > ori_w:
            padded_img = np.zeros((3, ori_h, ori_h), dtype='uint8')
            padded_foreground = np.zeros((ori_h, ori_h), dtype='uint8')
            padded_reprojected_lidar_depth = np.zeros((ori_h, ori_h), dtype='float64')
            padded_qry = np.zeros((ori_h, ori_h), dtype='float64')
            padded_out = np.zeros((ori_h, ori_h), dtype='float64')

            offset_w = ori_h // 2 - ori_w // 2

            padded_img[:,:,offset_w:offset_w + ori_w] = img
            padded_foreground[:,offset_w:offset_w + ori_w] = foreground
            padded_reprojected_lidar_depth[:,offset_w:offset_w + ori_w] = reprojected_lidar_depth
            padded_qry[:,offset_w:offset_w + ori_w] = tonumpy(qry)
            padded_out[:,offset_w:offset_w + ori_w] = tonumpy(out)

            K[1,2] = K[0,2]
          else:
            padded_img = np.zeros((3, ori_w, ori_w), dtype='uint8')
            padded_foreground = np.zeros((ori_w, ori_w), dtype='uint8')
            padded_reprojected_lidar_depth = np.zeros((ori_w, ori_w), dtype='float64')
            padded_qry = np.zeros((ori_w, ori_w), dtype='float64')
            padded_out = np.zeros((ori_w, ori_w), dtype='float64')

            offset_h = ori_w // 2 - ori_h // 2

            padded_img[:,offset_h:offset_h + ori_h,:] = img
            padded_foreground[offset_h:offset_h + ori_h,:] = foreground
            padded_reprojected_lidar_depth[offset_h:offset_h + ori_h,:] = reprojected_lidar_depth
            padded_qry[offset_h:offset_h + ori_h,:] = tonumpy(qry)
            padded_out[offset_h:offset_h + ori_h,:] = tonumpy(out)

            K[0,2] = K[1,2]

          img = padded_img
          foreground = padded_foreground
          reprojected_lidar_depth = padded_reprojected_lidar_depth
          qry = padded_qry
          out = padded_out

        else:
          img = crop_to_square(img)
          foreground = crop_to_square(foreground[None])[0]
          reprojected_lidar_depth = crop_to_square(reprojected_lidar_depth[None])[0]

      pcl = pixel2cam(totorch(reprojected_lidar_depth[None]),
                      totorch(K[None]))[0]
      normals = compute_normals_from_closest_image_coords(pcl[None, ...])[0]

      scene_things = dict(lidar_depth=tonumpy(reprojected_lidar_depth),
                          original_depth=tonumpy(qry),
                          refined_depth=tonumpy(out),
                          image=tonumpy(img),
                          intrinsics=tonumpy(K),
                          normals=normals,
                          foreground_mask=foreground)

      if not self.debug:
        os.makedirs(HNDR_CACHE_PATH, exist_ok=True)
        np.savez_compressed(cache_file, **scene_things)

    return scene_things

  def __getitem__(self, item):
    scene = self.scenes[item]
    scene_things = self.get_computed_depth_and_image(scene)

    img = scene_things['image']
    lidar_depth = scene_things['lidar_depth']
    original_depth = scene_things['original_depth']
    refined_depth = scene_things['refined_depth']
    K = scene_things['intrinsics']
    depth_mask = scene_things['foreground_mask']
    normals = scene_things['normals']

    dataset_info = dict()

    return img, \
           refined_depth, \
           depth_mask, \
           normals, \
           K, \
           dataset_info

if __name__ == '__main__':
  select_gpus('1')
  dataset = HNDRDataset('val', debug=False)
  dataset_elems = list(range(len(dataset)))
  random.shuffle(dataset_elems)
  for item_i in tqdm(dataset_elems):
    img, depth, depth_mask, normals, K, dataset_info = dataset[item_i]
    pcl = pixel2cam(totorch(depth[None]), totorch(K[None]))[0]

    imshow(img, title='img')
    imshow(normals, title='normals')
    imshow(depth, title='depth')
    show_pointcloud(pcl, img, title='pointcloud')
    show_pointcloud(pcl, img, valid_mask=depth_mask, title='masked_pointcloud')
