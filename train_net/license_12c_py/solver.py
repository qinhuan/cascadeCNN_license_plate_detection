#!/usr/bin/env python
# coding=utf-8

import caffe
from caffe.proto import caffe_pb2

Params = {
        # Train parameters
        'base_lr': 0.01,
        'weight_decay': 0.008,
        # 'lr_policy': "fixed",
        'lr_policy': "step",
        'stepsize': 30000,
        'gamma': 0.5,
        'momentum': 0.9,
        #'iter_size': 1,
        'max_iter': 400000,
        'snapshot': 10000,
        'display': 2500,
        #'average_loss': 40,
        'type': "SGD",
#        'solver_mode': None,
#        'device_id': None,
        #'debug_info': False,
        #'snapshot_after_train': True,
        # Test parameters
        'test_iter': 20,
        'test_interval': 2500,
        }



def create():
    #assert os.path.exists(os.path.dirname(Params['snapshot_prefix']))
    solver = caffe_pb2.SolverParameter(**Params)
    return solver
