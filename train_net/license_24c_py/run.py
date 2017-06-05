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
import license_24_net as Net 
import random
import cv2

data_dir = '/home/work/qinhuan/mywork/license_plate/data/lmdb'
train_lmdb = os.path.join(data_dir, 'license_train_12c_lmdb')
test_lmdb = os.path.join(data_dir, 'license_val_lmdb')
mean_file = os.path.join(data_dir, 'license_train_12c_imagenet_mean.binaryproto')
debug_iter = 2500
mbox_loss = []
x = []

#num_test_image = len([l for l in open(name_size_file, 'r').readlines()]) 
job_name = 'license_24c'
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
            'num_test_image': 1000,
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
    Solver.Params['base_lr'] = 0.002
    Solver.Params['test_iter'] = [Net.get_testiter()] 
    Solver.Params['test_interval'] = 500
    Solver.Params['stepsize'] = 30000
    Solver.Params['weight_decay'] = 0.004
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
    for test_it in range(20):
        solver.test_nets[0].forward()
        correct += sum(solver.test_nets[0].blobs['fc3'].data.argmax(1)
                                   == solver.test_nets[0].blobs['label'].data)
    test_acc.append(float(correct) / 1000)
    #print ('Iteration', it_num, 'Testing... accuracy=', float(correct) / 1000)
    plt.clf()
    plt.title('acc per 2500')
    plt.xlabel('Iter')
    plt.ylabel('accuracy')
    plt.plot(x, test_acc)
    plt.savefig(save_acc_path)

def rotate(img):
    height = img.shape[0]
    width = img.shape[1]
    rotateMat = cv2.getRotationMatrix2D((width/2, height/2, 180, 1))
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
    return rotateImg

def training(device_id=0):
    caffe.set_mode_gpu()
    caffe.set_device(device_id)
    solver_file = os.path.join(job_dir, 'solver.prototxt')
    sgd_solver = caffe.get_solver(solver_file)
    
    # if pretrain_model is not None:
    #     print('Finetune from {}.'.format(pretrain_model)) 
    #     sgd_solver.net.copy_from(pretrain_model)
    
    batch_size = 128
    
    # read train and test file list
    pos_list = []
    neg_list = []
    pos_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_pos_resize12x36.txt'
    neg_file = '/home/work/qinhuan/mywork/license_plate/data/data_list/all_neg24c_resize12x36.txt'
    f = open(pos_file, 'r')
    for line in f.readlines():
        pos_list.append(line.strip())
    f = open(neg_file, 'r')
    for line in f.readlines():
        neg_list.append(line.strip())
    print (len(pos_list), len(neg_list))
    test_list = []
    test_list[0:500] = pos_list[0:500]
    test_list[500:1000] = neg_list[0:500]
    pos_list = pos_list[500:]
    neg_list = neg_list[500:]
    print (len(pos_list), len(neg_list))
    random.shuffle(test_list)
    # pos_list_rotate = []
    # for line in pos_list:
    #     img_name = line.split(' ')[0]
    #     img = cv2.imread(img_name)

    pos_idx = [0]
    neg_idx = [0]
    HEIGHT = 12
    WIDTH = 36
    transformer = caffe.io.Transformer({'data': sgd_solver.net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([113, 96, 80]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    for i in range(Solver.Params['max_iter'] + 1):
        # feed_data(sgd_solver.net, start_train_idx, batch_size, train_files)
        feed_list = []
        num = batch_size / 2
        #print (pos_idx[0])
        #print (neg_idx[0])
        while num > 0:
            num -= 1
            feed_list.append(pos_list[pos_idx[0]])
            pos_idx[0] += 1
            if pos_idx[0] == len(pos_list):
                pos_idx[0] = 0
        num = batch_size / 2
        while num > 0:
            num -= 1
            feed_list.append(neg_list[neg_idx[0]])
            neg_idx[0] += 1
            if neg_idx[0] == len(neg_list):
                neg_idx[0] = 0
        random.shuffle(feed_list)
        for j in range(batch_size):
            line = feed_list[j].split(' ')
            img = caffe.io.load_image(line[0])
            img = transformer.preprocess('data', img)
            sgd_solver.net.blobs['data'].data[j, ...] = img
            sgd_solver.net.blobs['label'].data[j] = int(line[1])
        
        # sgd net
        sgd_solver.step(1)
        net = sgd_solver.net
        if debug_iter != -1:
            if (i + 1) % debug_iter == 0:
                debug(sgd_solver, True, i + 1)
            else:
                debug(sgd_solver, False, i + 1)

if __name__ == '__main__':
    #create_jobs()
    training(2)
