#/bin/bash

#!/bin/bash
mkdir -p datasets

wget -O datasets/google_scans.zip http://data.csail.mit.edu/synthetic_training/background_prompting/datasets/google_scans.zip
unzip datasets/google_scans.zip -d datasets/google_scans
rm datasets/google_scans.zip


