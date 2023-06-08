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


#!/usr/bin/env python
# Based on https://github.com/facebookresearch/moco/main_lincls.ply, to replicate DistributedDataParallel training
import resource
import os

import cv2
# to solve the error
# [ERROR:0@42.722] global /io/opencv/modules/core/src/parallel_impl.cpp (240) WorkerThread 12: Can't spawn new thread: res = 11
# following: https://stackoverflow.com/questions/71921941/joblib-not-working-rlimit-nproc-1-current-1-max
# cv2.setNumThreads(1)

# To solve the error:
# OpenBLAS blas_thread_init: RLIMIT_NPROC 2062360 current, 2062360 max
try:
    # to avoid ancdata error for too many open files, same as ulimit in console
    # maybe not necessary, but doesn't hurt
    resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
except:
    pass

import sys
sys.path.append('.')
sys.path.append('..')

import os

assert not '/afs/' in os.environ['HOME'], "Home should be set outside of afs directory"

import builtins
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import wandb

from utils import *

from object_prediction.depth_metrics_and_losses import *
from object_prediction.depth_datasets.object_dataset import ObjectDataset

from object_prediction.training_utils import *
from object_prediction.single_object_depth_predictors import get_model_and_transform

from collections import defaultdict

available_training_datasets = ['abo-renders', 'abo', 'hm3-abo', 'google-scans', 'nerf-sequences']
available_eval_datasets = [k for k in available_training_datasets] + ['hndr', 'dtu']

def get_train_args_parser():
  parser = argparse.ArgumentParser(description='PyTorch Object depth training')

  # dataset parameters
  parser.add_argument('--dataset-name', default='abo-renders', type=str, help='dataset to train on',
                      choices=available_training_datasets)
  parser.add_argument('--eval-datasets', default='hm3-abo,google-scans,nerf-sequences', type=comma_separated_str_list,
                      help='list of datasets to evaluate on (should not contain the training dataset), comma separated list of any of: {}'.format(','.join(available_training_datasets)))

  parser.add_argument('--eval-every-n-epochs', default=5, type=int,
                      help='list of datasets to evaluate on (should not contain the training dataset), comma separated list of any of: {}'.format(','.join(available_training_datasets)))

  parser.add_argument('--erode-masks', default='False', type=str2bool,
                      help='erode masks so that models that have uncertainty on the boundaries')
  parser.add_argument('--white-background', default='True', type=str2bool,
                      help='replace the background for a solid white texture')
  parser.add_argument('--randomize-configs-dataset', default="False", type=str2bool)

  parser.add_argument('--optimizer', default="sgd", type=str, choices=['sgd', 'adam'])

  parser.add_argument('--model', default='dpt_hybrid', choices=['midas_conv',
                                                                'dpt_hybrid',
                                                                'dpt',
                                                                'omnidata',
                                                                'leres_resnet50',
                                                                'leres_resnext101'],
                      help='model architecture (pretrained or not depending on --pretrained option on MiDaS) to use')
  parser.add_argument('--pretrained', default='True', type=str2bool,
                      help='model architecture (pretrained or not depending on --pretrained option on MiDaS) to use')

  parser.add_argument('--loss', default='si_rmse',
                      type=str, choices=['si_rmse', 'l1', 'l2'],
                      help='geometry loss to use')

  parser.add_argument('--normals-loss', default='cos_sim', type=str, choices=['none', 'cos_sim'],
                      help='model architecture (pretrained or not depending on --pretrained option on MiDaS) to use')


  parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                      help='number of data loading workers (default: 16)')
  parser.add_argument('--epochs', default=50, type=int, metavar='N',
                      help='number of total epochs to run')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                      help='manual epoch number (useful on restarts)')
  parser.add_argument('-b', '--batch-size', default=16, type=int,
                      metavar='N',
                      help='mini-batch size (default: 16), this is the total '
                           'batch size of all GPUs on the current node when '
                           'using Data Parallel or Distributed Data Parallel')

  parser.add_argument('--samples-per-epoch', default=10000, type=int)

  # we use 4 gpus by default
  parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                      metavar='LR', help='initial learnning rate, default on moco 0.015', dest='lr')
  parser.add_argument('--schedule', default=[40], nargs='*', type=int,
                      help='learning rate schedule (when to drop lr by 10x), default on moco [120,160] with 200 epochs')
  parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                      metavar='W', help='weight decay (default on moco: 1e-4)',
                      dest='weight_decay')
  parser.add_argument('-p', '--print-freq', default=10, type=int,
                      metavar='N', help='print frequen`cy (default: 10)')
  parser.add_argument('--plot-freq', default=200, type=int,
                      metavar='N', help='print frequency (default: 10)')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint (default: none)')
  parser.add_argument('--world-size', default=-1, type=int,
                      help='number of nodes for distributed training')
  parser.add_argument('--rank', default=-1, type=int,
                      help='node rank for distributed training')
  parser.add_argument('--dist-url', default='tcp://localhost:10035', type=str,
                      help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='nccl', type=str,
                      help='distributed backend')
  parser.add_argument('--seed', default=1337, type=int,
                      help='seed for initializing training.')
  parser.add_argument('--gpu', default=None, type=int,
                      help='GPU id to use when not using DataParallelDistributed and a single GPU.')
  parser.add_argument('--multiprocessing-distributed', action='store_true',
                      help='Use multi-processing distributed training to launch '
                           'N processes per node, which has N GPUs. This is the '
                           'fastest way to use PyTorch for either single node or '
                           'multi node data parallel training')

  parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

  parser.add_argument('--finetune-network', default='False', type=str2bool, help='finetune the network')
  parser.add_argument('--hyper-optimization', default='image', choices=['none', 'image', 'unet'],
                      type=str, help='optimize the background for objects')

  parser.add_argument('--unet-input', default="mask", choices=['mask', 'image'], type=str, help='optimize the background for objects')
  parser.add_argument('--add-bias-to-unet', default="True", type=str2bool, help='optimize the background for objects')

  parser.add_argument('--distortion-img-type', default='fft', type=str, choices=['fft', 'image'])
  parser.add_argument('--distortion-mode', default='background', type=str, choices=['additive', 'background'])

  # our custom args
  parser.add_argument('--result-folder', type=str, default='./results', help='Base directory to save model')
  parser.add_argument('--restart-latest', action='store_true', help='Restart latest checkpoint found in result_folder if it exists!')
  parser.add_argument('--store-all-checkpoints', type=str2bool, default='True', help='Whether to store all checkpoints or just the last one + best one')
  parser.add_argument('--log-wandb', type=str2bool, default="True", help='Whether to use Weights and Biases login')
  parser.add_argument('--restart-wandb', action='store_true', help='Restart latest checkpoint found in result_folder if it exists!')

  # debug args
  parser.add_argument('--fix-samples', default="False", type=str2bool)
  parser.add_argument('--n-samples-to-fix', default=1, type=int)
  parser.add_argument('--eval-only', default="False", type=str2bool)


  return parser

def main():
    print("Starting main process...")
    parser = get_train_args_parser()

    args = parser.parse_args()
    # assert args.distortion_mode == 'background', "Distortion mode should be background for all experiments until CVPR"

    for k in args.schedule:
      assert k < args.epochs, "--schedule should be epochs, so lower than maximum number of epochs (--epochs), {}/{}".format(k, args.epochs)

    for eval_dataset in args.eval_datasets:
      assert eval_dataset in args.eval_datasets, "Eval dataset {} not available!".format(eval_dataset)

    print("Will use dist url: " + args.dist_url)

    if args.fix_samples:
      assert 'debug' in args.result_folder, "Fix samples can only be used for debugging, and --result-folder does not have debug in path"

    args.save_folder = args.result_folder

    print("Will save on " + args.save_folder)
    os.makedirs(args.save_folder, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

global total_train_ii, total_val_ii, total_eval_ii
total_train_ii, total_val_ii = 0, 0
total_eval_ii = defaultdict(int)

def main_worker(gpu, ngpus_per_node, args):
    global total_train_ii, total_val_ii, total_eval_ii
    args.gpu = gpu
    print("Running process {}".format(gpu))

    debug = 'debug' in args.result_folder
    args.debug = debug
    if args.eval_only:
      assert args.debug, "Eval only should only be used when in debug mode."

    # suppress printing if not master
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        master = False
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        master = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print("Initializing distributed processes. If it gets stuck at this point, "
              "there may be another process running/zombie with same dist_url!")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("End initializing distributed processes.")
    # create model
    print("=> creating model '{}'".format(args.model))

    model, img_transform = get_model_and_transform(args)

    args.batch_size_total = args.batch_size

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # we set find_unused_parameters to avoid error of parameters that are not used
            # in the forward pass, as explained in ddp documentation:
            # https://pytorch.org/docs/stable/notes/ddp.html
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        debug = True
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported when using more than one gpu.")

    # define loss function (criterion) and optimizer
    assert args.loss == 'si_rmse', "Loss should be SI rmse in the final implementation if it"
    if args.loss == 'si_rmse':
      depth_criterion = si_rmse
    elif args.loss == 'l1':
      depth_criterion = l1
    elif args.loss == 'l2':
      depth_criterion = rmse

    assert args.finetune_network or args.pretrained, "--pretrained should be true except when --finetune-network is active (for training from scratch)"

    if hasattr(model, 'module'):
      model_instance = model.module
    else:
      model_instance = model

    parameters_to_opt = list(model_instance.get_trainable_parameters())
    total_parameters = sum([k.numel() for k in parameters_to_opt])
    assert args.finetune_network or args.hyper_optimization == 'unet' or total_parameters < 2e6, "Too many parameters ({}) being optimized with finetuning not activated. Check code!".format(total_parameters)
    print("Optimizing a total of {} parameters".format(total_parameters))
    if args.optimizer == 'sgd':
      optimizer = torch.optim.SGD(parameters_to_opt, args.lr, momentum=0.9)
    else:
      optimizer = torch.optim.Adam(parameters_to_opt, lr=args.lr, betas=(0.9, 0.99))

    if args.restart_latest:
        print("Trying to restart from latest checkpoint available")
        assert not args.resume, "A --resume checkpoint has been passed as argument, but restart_latest is also set (either one or the other should be set)!"
        # get latest checkpoint
        delete_all_checkpoints_except_last_loadable(args.save_folder)
        checkpoints = sorted([args.save_folder + '/' + k for k in listdir(args.save_folder, prepend_folder=False)
                              if k.startswith('checkpoint_') and k.endswith('.pth.tar')])
        while len(checkpoints) > 0:
            checkpoint = checkpoints.pop(-1)
            # test that checkpoint was properly saved
            if checkpoint_can_be_loaded(checkpoint):
              args.resume = checkpoint
              print("Restarting from checkpoint: " + checkpoint)
              break
            else:
              print("Checkpoint {} cannot be loaded! (Probably because of an error while storing)".format(checkpoint))

    best_val_loss = float('inf')
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            try:
              if args.gpu is None:
                  checkpoint = torch.load(args.resume)
              else:
                  # Map model to be loaded to specified single gpu.
                  loc = 'cuda:{}'.format(args.gpu)
                  checkpoint = torch.load(args.resume, map_location=loc)
              if debug:
                  # single gpu, remove module. from state_dict
                  final_state_dict = dict()
                  for k, v in checkpoint['state_dict'].items():
                      final_state_dict[k.replace('module.', '')] = v
                  checkpoint['state_dict'] = final_state_dict
            except Exception as e:
              print(e)
              print("Failed to load pickle: " + args.resume)

            args.start_epoch = checkpoint['epoch']

            total_train_ii = checkpoint['total_train_ii']
            total_val_ii = checkpoint['total_val_ii']
            total_eval_ii = checkpoint['total_eval_ii']

            best_val_loss = checkpoint['best_val_loss']

            if 'wandb_run_id' in checkpoint.keys() and not args.restart_wandb:
              wandb_run_id = checkpoint['wandb_run_id']
            elif args.log_wandb and not args.restart_wandb:
              print("Wandb loging enabled, but checkpoint does not have previous wandb_run_id!")

            if args.start_epoch >= args.epochs:
              print("All epochs already computed (desired max: {})".format(args.epochs))
              exit(0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(args.resume))
    else:
      print("No checkpoint resumed. Will start from scratch.")

    args.exp_name = 't-object-{}'.format(args.save_folder.split('/')[-1])
    ''' Wandb login + extra stuff to allow resume if job is killed'''
    assert len(args.exp_name) <= 128, "Wandb name should be <= 128 and is {}".format(len(args.exp_name))
    if master and args.log_wandb:
      if args.resume and not args.restart_wandb:
        resume_kwargs = dict(resume=wandb_run_id)
        print("Restarting wandb with id: " + wandb_run_id)
      else:
        wandb_run_id = args.exp_name + '_' + wandb.util.generate_id()
        print("Starting wandb with id: " + wandb_run_id)
        resume_kwargs = dict(id=wandb_run_id)
      wandb.init(project='object_depth', name=args.exp_name, **resume_kwargs)

      # replicate args to wandb config
      if not args.resume:
        for arg, argv in args.__dict__.items():
            wandb.config.__setattr__(arg, argv)
        if 'SLURM_JOB_ID' in os.environ.keys():
            wandb.config.__setattr__('SLURM_JOB_ID', os.environ['SLURM_JOB_ID'])
      wandb.config.update({'hostname': get_hostname()}, allow_val_change=True)

      # watch model to log parameter histograms/gradients/...
      wandb.watch(model)

    cudnn.benchmark = True

    print("Loading datasets")
    resolution = 448 if args.model.startswith('leres') else 384
    train_dataset = ObjectDataset(split='train',
                                  dataset_name=args.dataset_name,
                                  resolution=resolution,
                                  img_transform=img_transform,
                                  normalize_scale=True,
                                  erode_masks=args.erode_masks,
                                  background_type='white' if args.white_background else 'original',
                                  randomize_configs=args.randomize_configs_dataset)

    val_dataset = ObjectDataset(split='val',
                                dataset_name=args.dataset_name,
                                resolution=resolution,
                                img_transform=img_transform,
                                normalize_scale=True,
                                erode_masks=args.erode_masks,
                                background_type='white' if args.white_background else 'original',
                                randomize_configs=args.randomize_configs_dataset)

    eval_datasets = []
    for eval_dataset in args.eval_datasets:
      assert eval_dataset in available_eval_datasets, "Unknown eval dataset: {}".format(eval_dataset)
      if eval_dataset in ['nerf-sequences', 'hndr']:
        splits = ['val']
      else:
        # Use both train and val for eval
        splits = ['train', 'val']
      datasets = [ObjectDataset(split=split,
                                dataset_name=eval_dataset,
                                resolution=resolution,
                                img_transform=img_transform,
                                normalize_scale=True,
                                erode_masks=args.erode_masks,
                                background_type='white' if args.white_background else 'original',
                                randomize_configs=args.randomize_configs_dataset) for split in splits]
      eval_datasets.append(CombinedDataset(datasets))

    if args.fix_samples:
      assert args.n_samples_to_fix >= 1
      train_fixed_samples = [0] #[56756]  # sofa: [56756] chair that I always used during the internship: [1803715]
      train_fixed_samples = train_fixed_samples + random.sample(list(range(len(train_dataset))), args.n_samples_to_fix - len(train_fixed_samples))
      train_dataset = FixSampleDataset(train_dataset, replication_factor=args.samples_per_epoch, samples_to_fix=train_fixed_samples)
      val_dataset = FixSampleDataset(val_dataset, replication_factor=args.samples_per_epoch, samples_to_fix=list(range(args.n_samples_to_fix)))

      # eval_datasets = [FixSampleDataset(eval_dataset, replication_factor=args.samples_per_epoch, samples_to_fix=list(range(args.n_samples_to_fix))) for eval_dataset in eval_datasets]

    if args.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
      val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
      eval_samplers = [torch.utils.data.distributed.DistributedSampler(eval_dataset) for eval_dataset in eval_datasets]
    else:
      train_sampler = None
      val_sampler = None
      eval_samplers = [None for _ in eval_datasets]

    # set to True to avoid errors if we edit the sources while training. If set to False, python sources are reloaded at every epoch, so if there
    # when change to True, the following error started appearing (though could be that it was not a direct cause):
    # https://github.com/facebookresearch/detectron2/issues/3900
    # There are threads that are not killed and then give issues when one process dies
    persistent_workers = False
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers=persistent_workers, )
    val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
      num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True, persistent_workers=persistent_workers)
    eval_loaders = [torch.utils.data.DataLoader(
                    eval_dataset, batch_size=args.batch_size, shuffle=(eval_sampler is None),
                    num_workers=args.workers,
                    pin_memory=True, sampler=eval_sampler, drop_last=False, persistent_workers=persistent_workers) for (eval_dataset, eval_sampler) in zip(eval_datasets, eval_samplers)]

    print("Training with {} samples".format(len(train_dataset)))
    print("Validating with {} samples".format(len(val_dataset)))
    print("Evaluating with {} dataset with {} samples each".format(str(len(eval_loaders)),
                                                              ','.join([str(len(eval_dataset)) for eval_dataset in eval_datasets])))

    print("Datasets loaded. And starting training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and not train_sampler is None:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train and val for one epoch
        if not args.eval_only:
          train_rmse, train_losses = train_or_val(train_loader, model, depth_criterion, optimizer, epoch, args, master, mode='train')
          with torch.no_grad():
            val_rmse, val_losses = train_or_val(val_loader, model, depth_criterion, None, epoch, args, master, mode='val')

        eval_rmses = dict()
        eval_losses = dict()
        if epoch % args.eval_every_n_epochs == 0:
          with torch.no_grad():
            for eval_loader in eval_loaders:
              cur_dataset_name = eval_loader.dataset.datasets[0].dataset_name
              try:
                eval_rmse, eval_loss = train_or_val(eval_loader, model, depth_criterion, None, epoch, args, master, mode='eval', eval_dataset_name=cur_dataset_name)
                eval_rmses[cur_dataset_name] = float(eval_rmse.avg)
                eval_losses[cur_dataset_name] = float(eval_loss.avg)
                print("Eval loss for eval dataset {}: RMSE: {} Loss: {}".format(cur_dataset_name,
                                                                                eval_rmse.avg,
                                                                                eval_loss.avg))
              except Exception as e:
                print("Failed to evaluate dataset: {}. With exception: " + cur_dataset_name)
                print(e)

        if args.eval_only:
          exit()

        if val_rmse.avg < best_val_loss:
          is_best = True
          best_val_loss = val_rmse.avg
        else:
          is_best = False

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            info = {'epoch': epoch + 1,
                    'model': args.model,
                    'train_rmse': train_rmse.avg,
                    'train_loss': train_losses.avg,
                    'val_rmse': val_rmse.avg,
                    'val_loss': val_losses.avg,
                    'best_val_loss': best_val_loss,
                    'total_train_ii': total_train_ii,
                    'total_val_ii': total_val_ii,
                    'total_eval_ii': total_eval_ii,
                    'wandb_run_id': wandb_run_id,
                    'eval_rmses': eval_rmses,
                    'eval_losses': eval_losses}

            checkpoint_filename = args.save_folder + '/checkpoint_{:04d}.pth.tar'.format(epoch)
            print("Saving checkpoint in {}".format(checkpoint_filename))
            args_txt_file = args.save_folder + '/args.txt'
            args_pckl_file = args.save_folder + '/args.pckl'
            if not os.path.exists(args_txt_file):
              with open(args_txt_file, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            if not os.path.exists(args_pckl_file):
              dump_to_pickle(args_pckl_file, args)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                **info
            }, is_best=is_best, filename=checkpoint_filename)
            # also save info as a separate file for faster loading.
            if not args.store_all_checkpoints:
              delete_all_checkpoints_except_last_loadable(args.save_folder)
            print("Finished saving checkpoint!")
    print("Finished training!")
    # mark wandb as finished.
    wandb.finish(0)
    exit(1)

def train_or_val(train_loader, model, depth_criterion, optimizer, epoch, args, master, mode, eval_dataset_name=''):
    # if optimizer is None, it is train
    global total_train_ii, total_val_ii, total_eval_ii

    assert mode in ['train', 'val', 'eval'], "Mode {} not available!"

    is_training = mode == 'train'

    if mode == 'eval':
      prefix = 'eval_' + eval_dataset_name
    else:
      prefix = mode

    visdom_env = '{}_{}'.format(prefix, args.exp_name)

    if 'debug' in args.exp_name:
      visdom_env = 'PYCHARM_RUN'

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    rmses = AverageMeter('si-RMSE', ':6.2f')
    cos_sims = AverageMeter('Cos Sim', ':6.2f')
    if is_training:
      its_per_epoch = min(len(train_loader), args.samples_per_epoch / args.batch_size_total)
    else:
      # 10 times less
      its_per_epoch = min(len(train_loader), args.samples_per_epoch / args.batch_size_total / 10)
    its_per_epoch = int(its_per_epoch)
    progress = ProgressMeter(its_per_epoch,
                            [batch_time, data_time, losses, rmses, cos_sims],
                            prefix="[{}] Epoch: [{}]".format(mode[0].upper() + (':' + eval_dataset_name if mode == 'eval' else ''), epoch))

    # switch to train mode
    if is_training:
      model.train()
    else:
      model.eval()

    end = time.time()
    print("Starting {} epoch: {}".format(mode, epoch))

    for i, (imgs, object_masks, gt_depths, depth_masks, gt_normals, Ks, _, _, _) in enumerate(train_loader):
        if i > its_per_epoch:
          break

        '''
        for k, (img,depth,normals) in enumerate(zip(imgs, gt_depths, gt_normals)):
          imshow(depth, title='depth' + str(k))
          imshow(img, title='img' + str(k))
          imshow(normals, title='normals' + str(k))
        '''

        # measure data loading time
        cur_data_time = time.time() - end
        data_time.update(cur_data_time)

        bs = imgs.size(0)

        if args.gpu is not None:
          imgs = imgs.cuda(args.gpu, non_blocking=True)
          gt_depths = gt_depths.cuda(args.gpu, non_blocking=True)
          depth_masks = depth_masks.cuda(args.gpu, non_blocking=True)
          object_masks = object_masks.cuda(args.gpu, non_blocking=True)
          Ks = Ks.cuda(args.gpu, non_blocking=True)
        else:
          imgs = imgs.cuda()
          gt_depths = gt_depths.cuda()
          depth_masks = depth_masks.cuda()
          object_masks = object_masks.cuda()
          Ks = Ks.cuda()

        # compute output
        pred_depth, debug_info = model(imgs, object_masks)

        loss = depth_criterion(gt_depths, pred_depth, depth_masks)

        if args.normals_loss == 'cos_sim':
          pred_pcls = pixel2cam(pred_depth, Ks.cuda())
          pred_normals, normals_mask = compute_normals_from_closest_image_coords(pred_pcls, mask=depth_masks[:, None, :, :])
          normals_loss = cos_sim_loss(gt_normals[:,:,1:,1:].cuda(), pred_normals, normals_mask[:,0])
          loss += normals_loss
        else:
          pred_pcls = None
          pred_normals = None

        with torch.no_grad():
          cur_si_rmse = si_rmse(gt_depths, pred_depth, depth_masks)
          if pred_pcls is None:
            pred_pcls = pixel2cam(pred_depth, Ks.cuda())
            pred_normals, normals_mask = compute_normals_from_closest_image_coords(pred_pcls, mask=depth_masks[:,None,:,:])

          gt_normals = gt_normals.cuda()
          cur_cos_sim = compute_cos_sim(gt_normals[:,:,1:,1:], pred_normals, normals_mask[:,0]).mean()

          losses.update(loss.item(), bs)
          rmses.update(cur_si_rmse.item(), bs)
          cos_sims.update(cur_cos_sim.item(), bs)

        # compute gradient and do SGD step if training
        if is_training:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        # measure elapsed time
        cur_batch_time = time.time() - end
        batch_time.update(cur_batch_time)
        end = time.time()


        wandb_to_log = dict()
        if i % args.print_freq == 0:
            progress.display(i)

            if mode == 'train':
              iter_num = total_train_ii
            elif mode == 'val':
              iter_num = total_val_ii
            elif mode == 'eval':
              iter_num = total_eval_ii[eval_dataset_name]

            if args.log_wandb and master:
                wandb_to_log.update({prefix + '_loss/loss': float(loss.item()),
                                     prefix + '_loss/rmse': float(cur_si_rmse.item()),
                                     prefix + '_loss/cos_sim': float(cur_cos_sim.item()),
                                     prefix + '_iter_num': iter_num})

                if is_training:
                  wandb_to_log.update({'lr/lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                       'times/batch_time': cur_batch_time,
                                       'times/data_time': cur_data_time})


        if args.plot_freq != 0 and i % args.plot_freq == 0 and master:
          args_dict = dict(**args.__dict__)
          args_dict['hostname'] = get_hostname()
          visdom_dict(args_dict, title='args', env=visdom_env)

          if args.hyper_optimization != 'none':
            bg_image = debug_info['bg_image']
            visdom_histogram(bg_image, title='background values histogram', env=visdom_env)

            if len(bg_image.shape) == 4:
              # just select the first one
              if len(bg_image) == 1:
                bg_image = bg_image[0]
              else:
                bg_image = tile_images(bg_image)

            imshow(bg_image, title='learnt_background_image', env=visdom_env)

            bg_image_to_wandb = tonumpy(bg_image)
            bg_image_to_wandb = (bg_image_to_wandb - bg_image_to_wandb.min()) / (bg_image_to_wandb.max() - bg_image_to_wandb.min())
            bg_image_to_wandb = np.array(bg_image_to_wandb.transpose((1,2,0)) * 255.0, dtype='uint8')

            bg_image_wandb = wandb.Image(bg_image_to_wandb, caption="Background Image")

            wandb_to_log.update({"background-image": bg_image_wandb})

          try:
            gt_pcls = pixel2cam(gt_depths, Ks.cuda())

            for pcl_i in range(min(args.batch_size, 3)):
              img = tonumpy(imgs[pcl_i])
              img = (img - img.min()) / (img.max() - img.min())

              show_pointcloud(tonumpy(pred_pcls[pcl_i]), img, valid_mask=tonumpy(depth_masks[pcl_i]), title='pred_pcl_{}'.format(pcl_i), env=visdom_env)
              show_pointcloud(tonumpy(gt_pcls[pcl_i]), img, valid_mask=tonumpy(depth_masks[pcl_i]), title='gt_pcl_{}'.format(pcl_i), env=visdom_env)

              imshow(tonumpy(imgs[pcl_i]), title='img_{}'.format(pcl_i), env=visdom_env)
              imshow(tonumpy(pred_normals[pcl_i]), title='pred_normals_{}'.format(pcl_i), env=visdom_env)
              imshow(tonumpy(pred_normals[pcl_i] * depth_masks[pcl_i,:-1,:-1]), title='pred_normals_masked_{}'.format(pcl_i), env=visdom_env)
              imshow(tonumpy(gt_normals[pcl_i]), title='gt_normals_{}'.format(pcl_i), env=visdom_env)

              visdom_histogram(tonumpy(pred_depth[pcl_i][depth_masks[pcl_i] == 1]), title='pred_depth_values_{}'.format(pcl_i), env=visdom_env)
              visdom_histogram(tonumpy(gt_depths[pcl_i][depth_masks[pcl_i] == 1]), title='gt_depth_values_{}'.format(pcl_i), env=visdom_env)

              imshow(tonumpy(pred_depth[pcl_i] * object_masks[pcl_i]), title='depth_{}'.format(pcl_i), env=visdom_env)

              if pcl_i == 0:
                img_wandb = wandb.Image(np.array(255 * img.transpose((1, 2, 0)), dtype='uint8'),
                                        caption="Image")
                depth_to_wandb = tonumpy(pred_depth[pcl_i] * object_masks[pcl_i])
                depth_to_wandb = (depth_to_wandb - depth_to_wandb.min()) / (depth_to_wandb.max() - depth_to_wandb.min())

                depth_wandb = wandb.Image(np.array(depth_to_wandb * 255, dtype = 'uint8'), caption="Pred depth")

                wandb_to_log.update({"img": img_wandb})
                wandb_to_log.update({"depth": depth_wandb})

              if args.hyper_optimization != 'none':
                input_images = debug_info['model_inputs'][pcl_i]
                imshow(input_images, title='model_inputs_{}'.format(pcl_i), env=visdom_env)
                visdom_histogram(input_images, title='input_histogram', env=visdom_env)

          except Exception as e:
            print("Exception while plotting")
            print(e)
            pass

        if master and len(wandb_to_log) > 0 and args.log_wandb:
          wandb.log(wandb_to_log)

        if master:
          if mode == 'train':
            total_train_ii += 1
          elif mode == 'val':
            total_val_ii += 1
          elif mode == 'eval':
            total_eval_ii[eval_dataset_name] += 1
          else:
            raise Exception("Mode {} not present")

    return rmses, losses

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
      folder = '/'.join(filename.split('/')[:-1])
      if len(folder) > 0:
        folder += '/'
      shutil.copyfile(filename, folder + 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()