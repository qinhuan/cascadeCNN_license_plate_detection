from __future__ import print_function
#!/usr/bin/env python
# coding=utf-8
""" SSD """

import os, sys, math
import caffe
from caffe import layers as L
from caffe import params as P 
from caffe.proto import caffe_pb2 
Params = {
        'model_name': '',    
        'train_lmdb': '',
        'test_lmdb': '',
        'mean_file': '',
        'batch_size_per_device': 0,
        'test_batch_size': 0,
        'num_classes': 0,
        'num_test_image': 0,
}

train_transform_param = {}
test_transform_param = {}

def get_testiter():
    return Params['num_test_image'] / Params['test_batch_size'] 

def create_net(phase): 
    global train_transform_param
    global test_transform_param
    train_transform_param = {
            'mirror': False,
            'mean_file': Params['mean_file'] 
            }
    test_transform_param = {
            'mean_file': Params['mean_file'] 
            }
    if phase == 'train':
        lmdb_file = Params['train_lmdb']
        transform_param = train_transform_param
        batch_size = Params['batch_size_per_device']
    else:
        lmdb_file = Params['test_lmdb']
        transform_param = test_transform_param
        batch_size = Params['test_batch_size']

    net = caffe.NetSpec()
    if phase == 'test':
        net.data, net.label = L.Data(batch_size=batch_size,
            backend=P.Data.LMDB,
            source=lmdb_file,
            transform_param=transform_param,
            ntop=2) 
    elif phase == 'train':
        net.data = L.Input(shape=dict(dim=[128, 3, 6, 18]))
        net.label = L.Input(shape=dict(dim=[128]))
    elif phase == 'deploy':
        net.data = L.Input(shape=dict(dim=[1, 3, 6, 18]))
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.0001),
            'bias_filler': dict(type='constant')}
    net.conv1 = L.Convolution(net.data, num_output=16, kernel_size=3, **kwargs)
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    net.relu1 = L.ReLU(net.pool1, in_place=True)
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.005),
            'bias_filler': dict(type='constant')}
    net.fc2 = L.InnerProduct(net.pool1, num_output=16, **kwargs)
    net.relu2 = L.ReLU(net.fc2, in_place=True)
    net.drop2 = L.Dropout(net.fc2, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=100), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    net.fc3 = L.InnerProduct(net.fc2, num_output=2, **kwargs)
    if phase == 'train':
        net.loss = L.SoftmaxWithLoss(net.fc3, net.label)
    elif phase == 'test':
        net.accuracy = L.Accuracy(net.fc3, net.label)
    else:
        net.prob = L.Softmax(net.fc3)

    net_proto = net.to_proto()
    net_proto.name = '{}_{}'.format(Params['model_name'], phase)
    return net_proto

if __name__ == '__main__':
    for phase in ['train', 'test', 'deploy']:
        with open('/tmp/{}.prototxt'.format(phase), 'w') as f:
            print(create_net(phase), file=f)
