#/bin/bash

#!/bin/bash
mkdir -p datasets

wget -O datasets/nerfies.zip http://data.csail.mit.edu/synthetic_training/background_prompting/datasets/nerfies.zip
unzip datasets/nerfies.zip -d datasets/nerfies
rm datasets/nerfies.zip


