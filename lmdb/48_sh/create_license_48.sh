#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the 12 train + val data dirs
set -e

EXAMPLE=/home/work/qinhuan/mywork/license_plate/data/lmdb
DATA=/home/work/qinhuan/mywork/license_plate/data/data_list
TOOLS=/home/work/qinhuan/mywork/caffe/build/tools

TRAIN_DATA_ROOT=/home/work/qinhuan/mywork/license_plate/data/
VAL_DATA_ROOT=/home/work/qinhuan/mywork/license_plate/data/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=24
  RESIZE_WIDTH=72
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train_48c.txt \
    $EXAMPLE/license_train_48c_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val_48c.txt \
    $EXAMPLE/license_val_48c_lmdb

echo "Done."
