from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#!/usr/bin/env python
# coding=utf-8
import os, sys
import solver as Solver
import caffe
import numpy as np
import math
import license_48_net as Net 
import random
import cv2

data_dir = '/home/work/qinhuan/mywork/license_plate/data/lmdb'
train_lmdb = os.path.join(data_dir, 'license_train_48c_lmdb')
test_lmdb = os.path.join(data_dir, 'license_val_48c_lmdb')
mean_file = os.path.join(data_dir, 'license_train_48c_imagenet_mean.binaryproto')
debug_iter = 2500
mbox_loss = []
x = []

#num_test_image = len([l for l in open(name_size_file, 'r').readlines()]) 
job_name = 'license_48c_lmdb'
job_dir = '/home/work/qinhuan/mywork/license_plate/cascadeCNN_license_plate_detection/train_net/jobs/{}/'.format(job_name)
save_loss_path = os.path.join(job_dir, job_name + '_loss.png')
save_acc_path = os.path.join(job_dir, job_name + '_accuracy.png')

if not os.path.exists(job_dir):
    os.makedirs(job_dir)

def create_jobs():
    Net.Params = {
            'model_name': job_name,
            'train_lmdb': train_lmdb,
            'test_lmdb': test_lmdb,
            'mean_file': mean_file,
            'batch_size_per_device': 128,
            'test_batch_size': 50,
            'num_classes': 2,
            'num_test_image': 2000,
    }

    # train/test/deploy.prototxt
    for phase in ['train', 'test', 'deploy']:
        proto_file = os.path.join(job_dir, '{}.prototxt'.format(phase))
        with open(proto_file, 'w') as f:
            print(Net.create_net(phase), file=f)

    # solver.prototxt
    Solver.Params['train_net'] = os.path.join(job_dir, 'train.prototxt')
    Solver.Params['test_net'] = [os.path.join(job_dir, 'test.prototxt')]
    Solver.Params['snapshot_prefix'] = os.path.join(job_dir, job_name)
    Solver.Params['base_lr'] = 0.001
    Solver.Params['test_iter'] = [Net.get_testiter()] 
    Solver.Params['test_interval'] = 500
    Solver.Params['stepsize'] = 10000
    Solver.Params['weight_decay'] = 0.006
    Solver.Params['momentum'] = 0.9
    Solver.Params['max_iter'] = 400000
    Solver.Params['snapshot'] = 50000
    Solver.Params['display'] = 500
    Solver.Params['lr_policy'] = 'step'

    solver_file = os.path.join(job_dir, 'solver.prototxt')
    with open(solver_file, 'w') as f:
        print(Solver.create(), file=f)

test_acc = []
def debug(solver, flag, it_num):
    net = solver.net
    if len(mbox_loss) == 0:
        mbox_loss.append(0.0)
    mbox_loss[-1] = mbox_loss[-1] + float(net.blobs['loss'].data)
    if flag == False:
        return 
    if len(x) == 0:
        x.append(debug_iter)
    else:
        x.append(x[-1] + debug_iter)
    mbox_loss[-1] = mbox_loss[-1] / debug_iter
    plt.clf()
    plt.title('loss per 2500')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.plot(x, mbox_loss)
    plt.savefig(save_loss_path)
    mbox_loss.append(0.0)

    #print 'Iteration', it_num, 'testing...'
    correct = 0
    for test_it in range(Net.get_testiter()):
        solver.test_nets[0].forward()
        correct += sum(solver.test_nets[0].blobs['fc4'].data.argmax(1)
                                   == solver.test_nets[0].blobs['label'].data)
    test_acc.append(float(correct) / Net.Params['num_test_image'])
    #print ('Iteration', it_num, 'Testing... accuracy=', float(correct) / 1000)
    plt.clf()
    plt.title('acc per 2500')
    plt.xlabel('Iter')
    plt.ylabel('accuracy')
    plt.plot(x, test_acc)
    plt.savefig(save_acc_path)

def training(device_id=0):
    caffe.set_mode_gpu()
    caffe.set_device(device_id)
    solver_file = os.path.join(job_dir, 'solver.prototxt')
    sgd_solver = caffe.get_solver(solver_file)
    
    #transformer.set_mean('data', np.array([117, 103, 89]))

    for i in range(Solver.Params['max_iter'] + 1):
        sgd_solver.step(1)
        net = sgd_solver.net
        if debug_iter != -1:
            if (i + 1) % debug_iter == 0:
                debug(sgd_solver, True, i + 1)
            else:
                debug(sgd_solver, False, i + 1)

if __name__ == '__main__':
    create_jobs()
    training(3)
