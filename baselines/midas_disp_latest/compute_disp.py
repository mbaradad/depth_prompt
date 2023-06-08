from my_python_utils.common_utils import *
from my_python_utils.geom_utils import *

IMGS_FOLDER = '/data/vision/torralba/movies_sfm/projects/normals_acc/baselines/photo_3d_inpainting_facebook/image'

BOOST_BASE = '/data/vision/torralba/movies_sfm/projects/normals_acc/baselines/photo_3d_inpainting_facebook/BoostingMonocularDepth'

BOOST_INPUTS = 'cache/BoostingMonocularDepth_inputs'
BOOST_OUTPUTS = 'cache/BoostingMonocularDepth_outputs'


class BoostedMidasDispComputer():
  def __init__(self, dataset_name='no_dataset'):
    self.results_folder = '/data/vision/torralba/movies_sfm/home/scratch/mbaradad/caches/boosted_midas_outputs/' + dataset_name
    self.input_dir = 'cache/boosted_midas_inputs'

  def compute_midas_disp(self, img):
      img_hash = get_hash_from_numpy_array(img)
      # rm dir so that previous images don't get recomputed.
      os.makedirs(self.input_dir, exist_ok=True)
      os.makedirs(self.results_folder, exist_ok=True)
      cached_img_file = 'cache/boosted_midas_inputs/{}.png'.format(img_hash)
      cv2_imwrite(img, cached_img_file)
      return compute_boosting_disp(cached_img_file, self.results_folder)

def compute_boosting_disp(img_file, results_folder):
  # replicates boostmonodepth_utils.py from photo3d_inpainting_facebook
  output_file = results_folder + '/' + '.'.join(img_file.split('/')[-1].split('.')[:-1]) + '.npz'
  if os.path.exists(output_file):
    return np.load(output_file)['disp']

  boost_inputs_folder = BOOST_BASE + '/' + BOOST_INPUTS + '/' + img_file.split('/')[-1].split('.')[0]

  if os.path.exists(boost_inputs_folder):
    shutil.rmtree(boost_inputs_folder)

  os.makedirs(boost_inputs_folder, exist_ok=True)
  os.makedirs(BOOST_BASE + '/' + BOOST_OUTPUTS, exist_ok=True)

  base_name = os.path.basename(img_file)
  tgt_file = os.path.join(boost_inputs_folder, base_name)
  os.system(f'cp {img_file} {tgt_file}')

  # keep only the file name here.
  # they save all depth as .png file
  tgt_file = os.path.basename(tgt_file).replace('.jpg', '.png')

  if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    CUDA_STRING = 'CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES'])
  else:
    CUDA_STRING = ''

  command = f'{CUDA_STRING} cd {BOOST_BASE} && LD_LIBRARY_PATH=/data/vision/torralba/movies_sfm/home/anaconda3/lib /data/vision/torralba/movies_sfm/home/anaconda3/envs/default_env37/bin/python run.py --Final --data_dir ' \
            f'{boost_inputs_folder}/  --output_dir {BOOST_OUTPUTS} --depthNet 0'
  print("Running command: ")
  print(command)
  os.system(command)

  img = imageio.imread(img_file)
  H, W = img.shape[:2]
  scale = 640. / max(H, W)

  # resize and save depth
  target_height, target_width = int(round(H * scale)), int(round(W * scale))
  disp = imageio.imread(os.path.join(BOOST_BASE, BOOST_OUTPUTS, tgt_file).replace('.png.png', '.png'))
  disp = np.array(disp).astype(np.float32)

  disp = disp / 32768. - 1.

  np.savez_compressed(output_file, **dict_of(disp))

  return disp

from object_prediction.single_object_depth_predictors import get_dpt_transform

class MidasDispComputer():
  def __init__(self, model_type='DPT_Large'):
    self.model = torch.hub.load("intel-isl/MiDaS", model_type).cuda()
    self.transform = get_dpt_transform(model_type)

  def compute_midas_disp_batched(self, input_batch):
    with torch.no_grad():
      prediction = self.model(input_batch.cuda())

      prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=input_batch.shape[-2:],
        mode="bicubic",
        align_corners=False,
      )[:,0,:,:]

    disp_dpt = prediction

    assert disp_dpt.shape[1:] == input_batch.shape[-2:], "Shapes don't match"

    return disp_dpt

  def compute_midas_disp(self, img_file_or_img, device='cuda:0', biggest_image_dim=-1):
    if type(img_file_or_img) is str:
      img = cv2.imread(img_file_or_img)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      assert img_file_or_img.shape[0] == 3, "Should be passed with first dimension being channel, img.shape[0] == 3"
      img = img_file_or_img.transpose((1, 2, 0))

    if biggest_image_dim != -1:
      img = scale_image_biggest_dim(img.transpose((2, 0, 1)), biggest_image_dim).transpose((1, 2, 0))

    try:
      input_batch = self.transform(img).cuda()
    except:
      input_batch = totorch(self.transform({'image': img})['image'])[None,...].cuda()

    return self.compute_midas_disp_batched(input_batch).cpu().numpy()[0]


def compute_midas_disp(img_file_or_img, device, model_type='DPT_Large', biggest_image_dim=-1):
  computer = MidasDispComputer(model_type=model_type)

  disp_dpt = computer.compute_midas_disp(img_file_or_img, device, model_type='DPT_Large', biggest_image_dim=biggest_image_dim)

  return disp_dpt

def adjust_disp_photo3d_heuristic(disp, disp_rescale=3, blur=True):
  # extracted from photo_3d_inpainting.utils.read_MiDaS_depth, but modified a bit to ease debugging
  # TODO: add masking
  disp = disp - disp.min() + 5
  if blur:
    disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
  disp = (disp / disp.max()) * disp_rescale
  return disp

def disp_to_depth(disp, disp_rescale=3, blur=True):
  adjusted_disp = adjust_disp_photo3d_heuristic(disp, disp_rescale, blur)
  depth = 1. / adjusted_disp

  return depth

def depth_to_pcl(depth, K=None):
  depth = totorch(depth)
  if len(depth.shape) == 2:
    depth = depth[None]
    squeeze = True
  else:
    squeeze = False

  _, H, W = depth.shape

  if K is None:
    # from L 877 of photo_3d_inpainting.utils.py sdict['int_mtx'] = [...]
    K = totorch(np.array([[max(H, W), 0, W // 2],
                        [0, max(H, W), H // 2],
                        [0, 0, 1]]).astype(np.float32))
  else:
    K = totorch(K)

  if len(K.shape) == 2:
    K = K[None]

  pcls = tonumpy(pixel2cam(depth, K))
  return pcls[0] if squeeze else pcls


def disp_to_pcl(disp, K=None):
  assert type(disp) is np.ndarray
  depth = disp_to_depth(disp)
  return depth_to_pcl(depth, K)

if __name__ == '__main__':
  tgt_img = 'ai_001_010_scene_cam_00_final_hdf5_frame.0000_gt_depth.jpg'
  img_file = '{}/{}'.format(IMGS_FOLDER, tgt_img)
  boost_depth_file = img_file.replace('/image/', '/depth/').replace('.jpg', '.npy')

  disp_boost = np.load(boost_depth_file)
  gt_depth = np.load(boost_depth_file.replace('/depth/', '/gt_depths/'))
  gt_disp = 1 / gt_depth

  model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
  # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
  # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

  model = torch.hub.load("intel-isl/MiDaS", model_type)

  midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

  if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
  else:
    transform = midas_transforms.small_transform

  img = cv2.imread(img_file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  input_batch = transform(img).to()

  with torch.no_grad():
    prediction = model(input_batch)

    prediction = torch.nn.functional.interpolate(
      prediction.unsqueeze(1),
      size=img.shape[:2],
      mode="bicubic",
      align_corners=False,
    ).squeeze()

  disp_dpt = prediction.cpu().numpy()

  imshow(img, title='image')

  imshow(disp_dpt, title='dpt')
  imshow(disp_boost, title='boosting')
  imshow(gt_disp, title='gt')

  # same transformations as photo_3d_inpainting_facebook/utils.py, read_MiDaS_depth


  depth_dpt = totorch(disp_to_depth(disp_dpt))
  depth_boost = totorch(disp_to_depth(disp_boost))
  gt_depth = totorch(gt_depth)

  visdom_histogram(depth_boost, title='depth_boost')
  visdom_histogram(depth_dpt, title='depth_dpt')
  visdom_histogram(gt_depth, title='gt_depth')

  coords_boost = depth_to_pcl(depth_boost)
  coords_dpt = depth_to_pcl(depth_dpt)
  coords_gt = depth_to_pcl(gt_depth)

  show_pointcloud(coords_boost, img.transpose((2,0,1)), title='boost')
  show_pointcloud(coords_dpt, img.transpose((2,0,1)), title='dpt')
  show_pointcloud(coords_gt, img.transpose((2,0,1)), title='gt')

  exit(0)
