#!/usr/bin/env bash
GPU_0=$1
GPU_1=$2
re='^[0-9]+$'
if ! [[ $GPU_0 =~ $re ]] ; then
   echo "error: No GPU provided as argument" >&2; exit 1
fi
re='^[0-9]+$'
if ! [[ $GPU_1 =~ $re ]] ; then
   echo "error: No GPU provided as argument" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU_0,$GPU_1;

PORT=$(( $RANDOM % 40000 + 10000 ))
export DIST_URL="tcp://localhost:$PORT"
echo "Training with dist-url: $DIST_URL"

# 10 images fit in a 16gb gpu with dpt_hybrid, we put bs 40 as we train with 4 gpus
EXPERIMENT_NAME="single_background_fft_omnidata"
echo "Running experiment name: $EXPERIMENT_NAME"
# we set number of samples per epoch so that there are 150k iterations
# samples_per_epoch = batch_size * 150k iterations / 50 epochs
python object_prediction/train_object_model.py --workers 32 \
--batch-size 8 \
--epochs 50 \
--lr 5e-4 --schedule 40 \
--eval-every-n-epochs 4 \
--samples-per-epoch 24000 \
--loss si_rmse --normals-loss cos_sim --print-freq 10 --plot-freq 200 \
--dataset-name abo-renders \
--eval-datasets hndr,google-scans,nerf-sequences --eval-only False \
--dist-url $DIST_URL \
--world-size 1 --rank 0 \
--store-all-checkpoints True \
--fix-samples False \
--result-folder checkpoints/$EXPERIMENT_NAME \
--restart-latest --log-wandb True \
--multiprocessing-distributed \
--pretrained True \
--finetune-network False \
--randomize-configs-dataset False \
--white-background True \
--hyper-optimization image \
--distortion-img-type fft \
--distortion-mode background \
--add-bias-to-unet True \
--model omnidata
