#!/bin/bash
#SBATCH --job-name=imagenet_download
#SBATCH --partition=all
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=/playpen-shared/dinara/imagenet/download_%j.log

mkdir -p /playpen-shared/dinara/datasets/imagenet

# wget --user=YOUR_USER --password=YOUR_PASS \
#   https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar \
#   -P /playpen-shared/dinara/datasets/imagenet/

wget --user=adinara --password=xl1-2-3-4-5! \
  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar \
  -P /playpen-shared/dinara/datasets/imagenet/