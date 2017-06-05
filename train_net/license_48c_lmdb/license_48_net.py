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
    net.data, net.label = L.Data(batch_size=batch_size,
        backend=P.Data.LMDB,
        source=lmdb_file,
        transform_param=transform_param,
        ntop=2)
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            #'weight_filler': dict(type='gaussian', std=0.0001),
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant')}
    net.conv1_1 = L.Convolution(net.data, num_output=64, kernel_size=3, pad=1, **kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.conv1_1, num_output=64, kernel_size=3, pad=1, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)
    net.pool1 = L.Pooling(net.conv1_2, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=2)
    net.norm1 = L.LRN(net.pool1, local_size=5, alpha=1e-4, beta=0.75)
    net.conv2_1 = L.Convolution(net.norm1, num_output=64, kernel_size=3, pad=1, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.conv2_1, num_output=64, kernel_size=3, pad=1, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)
    net.pool2 = L.Pooling(net.conv2_2, pool=P.Pooling.MAX, kernel_size=3, pad=1, stride=2)
    net.norm2 = L.LRN(net.pool2, local_size=5, alpha=1e-4, beta=0.75)

    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            #'weight_filler': dict(type='gaussian', std=0.005),
            'bias_filler': dict(type='constant')}
    net.fc3 = L.InnerProduct(net.norm2, num_output=256, **kwargs)
    net.relu3 = L.ReLU(net.fc3, in_place=True)
    net.drop3 = L.Dropout(net.fc3, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=250), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            #'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}
    net.fc4 = L.InnerProduct(net.fc3, num_output=2, **kwargs)
    if phase == 'train':
        net.loss = L.SoftmaxWithLoss(net.fc4, net.label)
    elif phase == 'test':
        net.accuracy = L.Accuracy(net.fc4, net.label)
    else:
        net.prob = L.Softmax(net.fc4)

    net_proto = net.to_proto()
    if phase == 'deploy':
        del net_proto.layer[0]
        net_proto.input.extend(['data'])
        net_proto.input_dim.extend([1,3,24,72])
    net_proto.name = '{}_{}'.format(Params['model_name'], phase)
    return net_proto

if __name__ == '__main__':
    for phase in ['train', 'test', 'deploy']:
        with open('/tmp/{}.prototxt'.format(phase), 'w') as f:
            print(create_net(phase), file=f)
