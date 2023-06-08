# Background Prompting for Improved Object Depth

Official implementation of Background Prompting for Improved Object Depth. This is not an officially supported Google product.
<p align="center">
  <img width="100%" src="./assets/gif_teaser.gif">
</p>

Background prompting is a simple but yet effective strategy that adapts the input object image with a learned background to improve object depth. We learn the background prompts only using small scale synthetic object datasets. To infer object depth on a real image, we place the segmented object into the learned background prompt and run off-the-shelf depth networks. Background Prompting helps the depth networks focus on the foreground object, as networks are made invariant to background variations. Moreover, Background Prompting minimizes the domain gap between synthetic and real object images, leading to better sim2real generalization than simple finetuning.

[[Project page](https://mbaradad.github.io/depth_prompt)]
[[arXiv](https://arxiv.org/abs/23)]
[[Paper](https://arxiv.org/pdf/23)]

## Requirements
The code has been tested with Python 3.7.12 and Pytorch 1.13.0.

```
conda create -n depth_prompt python=3.7.12
pip install -r requirements.txt
```
You will also need to install opencv, for example with conda with:
```
conda install -c conda-forge opencv
```
To train or evaluate prompting for LeReS, you will need to create a LeReS environment 
following the instructions in the original repo, and then install again the previous requirements.

## Training

### Downloading datasets
For repeating the training in the dataset, it is necessary to obtain the rendered sequences of both 
[ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) 
and
[HM3D-ABO](https://github.com/zhenpeiyang/HM3D-ABO)
datasets from the original sources. You can train with a subset of this (or the other available datasets) using the argument `--dataset-name` in the training script. See scripts under `scripts/` for examples


### Training
After downloading the datasets, you can train each of the using the scripts contained in the folder `scripts/train`.
For example, to train a single background for DPT, you can use the following command:

```
python object_prediction/train_object_model.py --workers 32 --batch-size 8 \
--epochs 50 --lr 5e-4 --schedule 40 --samples-per-epoch 24000 \
--dataset-name abo-renders \
--eval-datasets hndr,google-scans,nerf-sequences --eval-only False \
--dist-url $DIST_URL \
--world-size 1 --rank 0 \
--result-folder checkpoints/$EXPERIMENT_NAME \
--restart-latest --log-wandb True \
--multiprocessing-distributed \
--pretrained True \
--finetune-network False \
--white-background True \
--hyper-optimization image \
--distortion-img-type fft \
--distortion-mode background \
--add-bias-to-unet True \
--model dpt
```

## Evaluation

### Downloading models
We provide pretrained models for both single background and hypernet, which can be downloaded using:
```
./scripts/downloads/pretrained_models.sh
```

To evaluate, you will need to download the rest of the datasets (Google Scans, HNDR, and Nerf Sequences) and the pretrained models.
To download Nerf Sequences, you can use the script under `scripts/downloads/nerf_sequences.sh`. For the rest of the datasets, 
use the respective downloads for each of the datasets.

To download our pretrained models, you can use the script under:
`scripts/pretrained_models.sh`