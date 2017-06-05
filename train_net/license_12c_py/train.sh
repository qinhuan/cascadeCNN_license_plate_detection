#!/bin/sh
# setting caffe root folder
CAFFE_ROOT=/home/work/qinhuan/mywork/caffe
CURRENT_DIR=`pwd`
# add pycaffe path
export PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python:$CURRENT_DIR

nohup python ./run.py >log.txt 2>&1 &
#python ./run.py
