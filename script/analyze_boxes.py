#!/usr/bin/python
# -*- coding: utf-8 -*-
#analysis data distribution, include w h ox oy w/h， and draw histogram
from xml.dom.minidom import Document

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import pdb
import sys, os

global num
global num_y
def count(split_lines, img_size, class_ind, total):
    width = img_size[1]
    height = img_size[0]
    
    global num
    global num_y
    for split_line in split_lines:
        line = split_line.strip().split(' ')
        xmin = float(line[0])
        ymin = float(line[1])
        xmax = float(line[2]) + xmin
        ymax = float(line[3]) + ymin
        w = xmax - xmin
        h = ymax - ymin
        ox = (xmin + xmax) / 2.0
        oy = (ymin + ymax) / 2.0
        num += 1
        if oy / height < 0.5:
            num_y += 1
        # print xmin,xmax,ymin,ymax,w,h,ox,oy
        total['license_plate']['w'].append(w / width)
        total['license_plate']['h'].append(h / height)
        total['license_plate']['WW'].append(w)
        total['license_plate']['HH'].append(h)
        total['license_plate']['ox'].append(ox / width)
        total['license_plate']['oy'].append(oy / height)
        total['license_plate']['aspect_ratio'].append(w / h)

def save_hist(total, hist_dir):
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)
    
    for cls, lists in total.items():
        for pro, lis in lists.items():
            plt.clf()
            params = {'bins':100}
            plt.hist(lis, **params)
            plt.savefig(os.path.join(hist_dir, '{}_{}.jpg'.format(cls, pro)))
            

if __name__ == '__main__':
    
    global num 
    global num_y 
    num = 0
    num_y = 0
    class_ind = ['license_plate']
    prop_idx = ('WW', 'HH', 'w', 'h', 'ox', 'oy', 'aspect_ratio')
    total = {}
    for c in class_ind:
        total[c] = {} 
        for p in prop_idx:
            total[c][p] = []
    
    hist_dir = ('hist')
    txts_dir = '/home/work/qinhuan/mywork/license_plate/data/license_plate_txts'
    for file in os.listdir(txts_dir):
        if file.endswith('.txt'):
            with open(txts_dir + '/' + file, 'r') as f:
                split_lines = []
                for line in f.readlines():
                    split_lines.append(line)
                img_size = (836, 2732, 3)
                count(split_lines, img_size, class_ind, total)
    
    save_hist(total, hist_dir)
    print num, num_y
