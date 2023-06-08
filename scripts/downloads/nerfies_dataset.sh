#/bin/bash

#!/bin/bash
mkdir -p encoders

encoders=('single_background_fft_midas_conv.pth.tar' \
          'single_background_fft_leres_resnet50.pth.tar' \
          'single_background_fft_omnidata.pth.tar' \
          'single_background_fft_dpt.pth.tar' \

          'hypernet_fft_midas_conv.pth.tar' \
          'hypernet_fft_leres_resnet50.pth.tar' \
          'hypernet_fft_omnidata.pth.tar' \
          'hypernet_fft_dpt.pth.tar' \
           )

for ENCODER in ${encoders[@]}
do
    echo "Downloading $ENCODER"
    wget -O encoders/$ENCODER http://data.csail.mit.edu/synthetic_training/background_prompting/models/$ENCODER
done




