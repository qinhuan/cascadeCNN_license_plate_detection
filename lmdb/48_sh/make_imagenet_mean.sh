#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/work/qinhuan/mywork/license_plate/data/lmdb
DATA=/home/work/qinhuan/mywork/license_plate/data/lmdb
TOOLS=/home/work/qinhuan/mywork/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/license_train_48c_lmdb \
  $DATA/license_train_48c_imagenet_mean.binaryproto

echo "Done."
