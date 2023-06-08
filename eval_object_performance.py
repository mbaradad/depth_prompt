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


from utils import *
import sys

sys.path.append('.')

parser = argparse.ArgumentParser(description='Predict single object')
struct_methods_to_type = dict([('object_prediction', 'depth'), # our final models and finetuned ones the rest are baselines

                               ('dpt', 'disp'),
                               ('dpt_hybrid', 'disp'),
                               ('midas_conv', 'disp'),
                               ('boosted_midas', 'disp'),
                               ('leres_resnet50', 'depth'),
                               ('leres_resnext101', 'depth'),
                               ('omnidata', 'depth'),
                               ('planar_prediction', 'depth')])

parser.add_argument('--struct-method',
                    default='object_prediction',
                    choices=struct_methods_to_type.keys(),
                    help='what structure method to use')
parser.add_argument('--dataset', default='hndr',
                    type=str,
                    choices=['abo',
                             'hm3-abo',
                             'abo-renders',
                             'google-scans',
                             'nerf-sequences',
                             'hndr',
                             'dtu',
                             'ners',
                             'lego-mobile',
                             'image-folder'],
                    help='what dataset to test')

parser.add_argument('--image-folder', default='/data/vision/torralba/movies_sfm/home/normals_acc/test_images', type=str, help='image folder to test')

parser.add_argument('--dump-results-only', default='False', type=str2bool)

parser.add_argument('--checkpoint',
                    default='/data/vision/torralba/movies_sfm/home/normals_acc/object_prediction/checkpoints/11_02_single_background_fft_dpt_lr1e-3_schedule40/model_best.pth.tar',
                    type=str,
                    help='checkpoint file to use if it is not a baseline method')

parser.add_argument('--split', default='val', type=str, help='select split, for debugging purposes', choices=['train', 'val', 'all'])

parser.add_argument('--fit-disp-baseline', default="False", type=str2bool)
parser.add_argument('--background-type', default="white", type=str, choices=['white', 'original', 'random_noise', 'random_texture'])

parser.add_argument('--print-freq', default=1, type=int)
parser.add_argument('--debug-plot-freq', default=1, type=int)

parser.add_argument('--results-folder', default='test_object_performance', type=str)

parser.add_argument('--workers', default=10, type=int)
parser.add_argument('--batch-size', default=1, type=int)

parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--debug', default="True", type=str2bool)

parser.add_argument('--n-samples-to-test', default=300, type=int)
parser.add_argument('--n-samples-to-store', type=int, default=100)
parser.add_argument('--seed', default=1338, type=int, help='seed for initializing training.')

parser.add_argument('--random-sorted-samples', default="True", type=str2bool, help='process samples at random, so that stored predictions are diverse')

args = parser.parse_args()
print("Eval with args:")

select_gpus(args.gpu)
if args.seed is not None and args.seed != -1:
  # fix seed so that
  np.random.seed(args.seed)
  random.seed(args.seed)
  torch.manual_seed(args.seed)

assert not args.split == 'train' or args.debug, "Train split should only be called for debugging!"

import torch.utils.data.sampler

from utils import *

from object_prediction.depth_datasets.object_dataset import ObjectDataset

if get_conda_env() != 'LeReS':
  from object_prediction.single_object_depth_predictors import get_model_and_transform
  from baselines.omnidata_main.omnidata_tools.torch.depth_and_normals_predictor import OmnidataDepthAndNormalsPredictor
  from baselines.midas_disp_latest.compute_disp import MidasDispComputer, BoostedMidasDispComputer, adjust_disp_photo3d_heuristic
  from object_prediction.single_object_depth_predictors import DPTDepthComputer

from dpt.my_midas_utils import *

from object_prediction.depth_metrics_and_losses import si_rmse, compute_cos_sim, recover_scale_depth

from object_prediction.training_utils import AverageMeter, ProgressMeter

class PlanarDepthPredictor:
  def __init__(self):
    self.transform = None
    return

  def compute_depth(self, img, mask=None):
    return np.ones((img.shape[-2:]))

  def compute_depth_batched(self, img, masks=None):
    return totorch(np.ones((img.shape[0], *img.shape[-2:])))

if __name__ == '__main__':
  os.makedirs(args.results_folder, exist_ok=True)

  results_folder = args.results_folder
  performance_file = results_folder + '/performance_dataset_{}_method_n_test_samples_{}'.format(args.dataset, args.n_samples_to_test)
  assert args.background_type in ['white', 'original', 'random_noise', 'random_texture']
  if args.background_type == 'white':
    performance_file += '_white_background'
  elif args.background_type == 'original':
    performance_file += '_original_background'
  else:
    performance_file += '_{}_background'.format(args.background_type)

  if os.path.exists(performance_file) and not args.debug and not args.dump_results_only:
    print("Performance file already exists in: {}. Won't compute again evaluation!".format(performance_file))
    exit(0)
  elif args.dump_results_only:
    print("Will dump results only!!!!!!!!!")
  else:
    print("Performance file not found, will compute!: {}".format(performance_file))

  os.makedirs(results_folder, exist_ok=True)

  struct_prediction_type = struct_methods_to_type[args.struct_method]
  struct_method_name = args.struct_method

  img_transform = None
  training_args = None

  ## MODEL DEFINITION ##

  # all baselines that we test
  if args.struct_method == 'omnidata':
    disp_or_depth_computer = OmnidataDepthAndNormalsPredictor('depth')
  elif args.struct_method.startswith('leres'):
    from baselines.leres.LeReS.Minist_Test.tools.depth_predictor import LeresDepthPredictor
    backbone_name = args.struct_method.split('_')[1]
    disp_or_depth_computer = LeresDepthPredictor(backbone_name)
  elif args.struct_method == 'boosted_midas':
    MAX_BOOSTED_MIDAS_SAMPLES = 1000
    assert args.dump_results_only or args.n_samples_to_test <= MAX_BOOSTED_MIDAS_SAMPLES, "Boosted midas takes long to run per sample (and results are stored), " \
                                         "decrease the number of samples or remove the assert ({} max:{})".format(args.n_samples_to_test, MAX_BOOSTED_MIDAS_SAMPLES)
    disp_or_depth_computer = BoostedMidasDispComputer(args.dataset)
  elif args.struct_method == 'planar_prediction':
    disp_or_depth_computer = PlanarDepthPredictor()
  elif args.struct_method in ['dpt', 'dpt_hibrid', 'midas_conv']:
    model_types = {'dpt': 'DPT_Large',
                   'dpt_hibrid': 'DPT_Hybrid',
                 'midas_conv': 'MiDaS'}
    model_type = model_types[args.struct_method]
    disp_or_depth_computer = MidasDispComputer(model_type)
  # the last one is ours
  elif args.struct_method.startswith('object_prediction'):
    args_pickle_file = '/'.join(args.checkpoint.split('/')[:-1]) + '/args.pckl'
    training_args = load_from_pickle(args_pickle_file)
    from object_prediction.train_object_model import get_train_args_parser
    parser = get_train_args_parser()
    defaults = vars(parser.parse_args([]))
    for k, v in defaults.items():
      if not k in training_args.__dict__.keys():
        print("Missing arg --{} on original training args, will use default: {}".format(k, v))
        training_args.__dict__[k] = v
    disp_or_depth_computer, img_transform = get_model_and_transform(training_args)

    state_dict = torch.load(args.checkpoint)['state_dict']
    state_dict_to_load = dict()
    for k, v in state_dict.items():
      state_dict_to_load[k.replace('module.', '')] = v
    disp_or_depth_computer.load_state_dict(state_dict_to_load)
    disp_or_depth_computer = disp_or_depth_computer.cuda()

  if img_transform is None and args.struct_method not in ['boosted_midas']:
    assert hasattr(disp_or_depth_computer, 'transform'), "Disp or depth computer should have transform " \
                                                         "when predicting on whole dataset. So that is computed by the workers."
    img_transform = disp_or_depth_computer.transform

  ## END MODEL DEFINITION ##

  assert not disp_or_depth_computer is None, "Struct method {} not valid!".format(args.struct_method)

  # we normalize scale so that small or big objects have mean 1, penalize the same towards the final loss.
  resolution = 448 if args.struct_method == 'object_prediction' and 'leres' in args.checkpoint else 384
  dataset = ObjectDataset(args.split,
                          resolution=resolution,
                          dataset_name=args.dataset,
                          img_transform=img_transform,
                          erode_masks=False,
                          background_type=args.background_type,
                          normalize_scale=True,
                          image_folder=args.image_folder,
                          zero_pad=True)


  # use seed so that the samples plotted and tested are the same across datasets
  n_samples_to_test = min(len(dataset), args.n_samples_to_test)
  loader_sampler = ListSampler(dataset, n_samples_to_test, seed=args.seed, debug=args.debug, shuffle_samples=True)

  os.makedirs(results_folder, exist_ok=True)

  loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=loader_sampler, drop_last=False)

  pbar = tqdm(total=n_samples_to_test)
  si_rmse_meter = AverageMeter('si-RMSE', ':6.2f')
  cossim = AverageMeter('Cos Sim', ':6.2f')

  progress = ProgressMeter(len(loader),
                           [si_rmse_meter, cossim],
                           prefix="method: {}, its:".format(args.struct_method))

  n_processed = 0
  n_stored_preds = 0
  debug_info = None
  for batch_i, (imgs, object_masks, gt_depths, depth_masks, gt_normals, Ks, _, _, _) in enumerate(loader):
    if struct_prediction_type == 'depth':
      # DEPTH PREDICTORS
      try:
        depths = disp_or_depth_computer.compute_depth_batched(imgs.cuda(), masks=object_masks.cuda())
        if type(depths) is tuple:
          depths, debug_info = depths
      except Exception as e1:
        try:
          if 'CUDA' in str(e1):
            raise e1
          depths, debug_info = disp_or_depth_computer.get_depth(imgs, object_masks=object_masks)
        except Exception as e2:
          # trying non batched
          try:
            if 'CUDA' in str(e2):
              raise e2
            depths = []
            for i, img in enumerate(imgs):
              with torch.no_grad():
                cur_depth = disp_or_depth_computer.compute_depth(img, mask=object_masks[i])
              cur_depth = tonumpy(cur_depth)
              depths.append(totorch(cur_depth)[None])
            depths = torch.cat(depths).cuda()
          except Exception as e3:
            print("All methods for obtaining depth failed, wiht exceptions:")
            print(e1)
            print(e2)
            print(e3)
    else:
      # DISP PREDICTORS, MIDAS AND BOOSTED MIDAS
      try:
        disp = disp_or_depth_computer.compute_midas_disp_batched(imgs)
      except:
        # process one by one, for boosted_midas for example, which does not implement batched
        disps = []
        for _, img in enumerate(imgs):
          with torch.no_grad():
            cur_disp = disp_or_depth_computer.compute_midas_disp(np.array(img))
          cur_disp = tonumpy(cur_disp)
          disps.append(totorch(cur_disp)[None])
        disp = torch.cat(disps).cuda()
      if args.fit_disp_baseline:
        midas_constants = MidasConstants(Ks)

        gt_disps = np.array(gt_depths)
        gt_disps[object_masks != 1] = 1e-4
        gt_disps = 1 / gt_disps
        gt_disps[object_masks != 1] = 0

        assert np.isinf(tonumpy(gt_disps)).sum() == 0, "Some inf detedted in gt disparity after fidding disparity baseline!"

        disp_adjusted = midas_constants.initialize_with_fit(disp, gt_disps, mask0=depth_masks, robust=False)
        disp_adjusted[depth_masks == 0] = 1
        depths = totorch(1 / disp_adjusted).cuda()
        depths[depth_masks == 0] = 0
        depths[depths != depths] = 0
        assert np.isinf(tonumpy(depths)).sum() == 0, "Some inf detedted after fidding disparity baseline!"
        assert np.isnan(tonumpy(depths)).sum() == 0, "Some nan detedted after fidding disparity baseline!"
      else:
        # use photo_3d_inpainting strategy
        # extracted from photo_3d_inpainting.utils.read_MiDaS_depth, but modified a bit to ease debugging
        disp = disp - disp.min(-1)[0].min(-1)[0][:,None,None]
        max_disps = disp.max(-1)[0].max(-1)[0][:,None,None]
        #disp = cv2.blur(tonumpy(disp / max_disps), ksize=(3, 3)) * tonumpy(max_disps)
        disp = tonumpy(disp / max_disps)
        depths = totorch(1. / np.maximum(disp, 0.05)).cuda()

    depths = depths.cuda()

    has_gt = gt_depths.shape[1] != 1
    if has_gt:
      cur_si_rmse = si_rmse(gt_depths.cuda(), depths.cuda(), depth_masks.cuda(), return_per_sample=True)
      scaled_pred_depth, scale_batch = recover_scale_depth(depths[:, None].cuda(), gt_depths[:, None].cuda(), depth_masks[:, None].cuda())
      scaled_pred_depth = scaled_pred_depth[:,0]

    else:
      scaled_pred_depth = depths

    pcls = tonumpy(pixel2cam(scaled_pred_depth, Ks.cuda()))

    pred_normals, normals_pred_mask = compute_normals_from_closest_image_coords(pcls,
                                                                                depth_masks[:, None])
    normals_pred_mask = normals_pred_mask[:,0]

    if has_gt:
      cur_cos_sim = compute_cos_sim(totorch(pred_normals), totorch(gt_normals[:, :, 1:, 1:]), normals_pred_mask)

      if cur_si_rmse.mean().item() != cur_si_rmse.mean().item():
        print("NaN detected while computing results for result folder: {}, will exit!".format(args.results_folder))
        exit(0)

      si_rmse_meter.update(cur_si_rmse.mean().item(), imgs.shape[0])
      cossim.update(cur_cos_sim.mean().item(), imgs.shape[0])

    pbar.update(imgs.shape[0])
    depths = tonumpy(depths)
    if batch_i % args.print_freq == 0:
      progress.display(batch_i)

    if n_stored_preds < args.n_samples_to_store :
      stored_preds_folder = results_folder + '/dumped_example_results'
      os.makedirs(stored_preds_folder, exist_ok=True)
      for i, (image, depth, depth_mask, K) in enumerate(zip(tonumpy(imgs), tonumpy(depths), tonumpy(depth_masks), tonumpy(Ks))):
        sample_i = batch_i * args.batch_size + i
        denormalized_img = (image - image.min())/(image.max() - image.min())

        things_to_save = dict(depth=depth,
                              depth_mask=depth_mask,
                              image=(image + 1) * 127.5,
                              K=K)
        if not debug_info is None:
          for k,v in debug_info.items():
            things_to_save[k] = v[i]

        if has_gt:
          things_to_save['gt_depth'] = gt_depths[i]

        file_to_save = '{}/{}.npz'.format(stored_preds_folder, str(sample_i).zfill(6))
        np.savez_compressed(file_to_save, **things_to_save)
        n_stored_preds += 1

        if n_stored_preds >= args.n_samples_to_store:
          break

    n_processed += imgs.shape[0]
    if n_processed > n_samples_to_test:
      break
    if n_stored_preds >= args.n_samples_to_store and args.dump_results_only:
      print("Sotred all desired preds, and --dump-results-only is True. Will exit now!")
      exit(0)


    if batch_i % args.debug_plot_freq == 0 and args.debug:
      for i_plt in range(0, len(imgs)):
        try:
          env = 'test_object_performance_' + struct_method_name


          denormalized_img_0 = (imgs[i_plt] - imgs[i_plt].min())/(imgs[i_plt].max() - imgs[i_plt].min())
          show_pointcloud(pcls[i_plt], denormalized_img_0, title='predicted_pcl', valid_mask=object_masks[i_plt], env=env)

          try:
            pred_video = render_pointcloud(pcls[i_plt], denormalized_img_0, valid_mask=object_masks[i_plt], add_camera_frustrum=True, K=Ks[i_plt])
          except Exception as e:
            print("Exception while printing. Maybe you want to set the following enrivonment variable:")
            print("LD_LIBRARY_PATH=/data/vision/torralba/movies_sfm/home/anaconda3/lib")
            print(e)

          temp_name = '{}/{}.gif'.format(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
          pred_video, _, _ = preprocess_im_to_plot(pred_video, normalize_image=True)
          make_gif(pred_video, path=temp_name, fps=10, biggest_dim=None)

          if has_gt:
            try:
              gt_pcls = tonumpy(pixel2cam(gt_depths.cuda(), Ks.cuda()))
              gt_video = render_pointcloud(gt_pcls[i_plt], denormalized_img_0, valid_mask=object_masks[i_plt], add_camera_frustrum=True, K=Ks[i_plt])
              mixed_video = []
              FRAMES_PER_TYPE = 5
              for i in range(len(gt_video)):
                if (i // FRAMES_PER_TYPE % 2) == 0:
                  mixed_video.append(pred_video[i])
                else:
                  mixed_video.append(gt_video[i])
            except:
              pass

        except Exception as e:
          print(e)
          print("K: " + str(Ks[i_plt]))
          pass
    del depths
  pbar.close()

  if not args.debug and not args.dump_results_only:
    performance_lines = ["Dataset: " + args.dataset, "Method {}, with {} samples".format(struct_method_name, n_samples_to_test)]
    for m in progress.meters:
      performance_lines.append('{}: {}'.format(m.name, m.avg))

    for l in performance_lines:
      print(l)

    write_text_file_lines(performance_lines, performance_file)