cascadeCNN_license_plate_detection
======================================
Implement cascade cnn for license plate detection
****
### Author: HuanQin
### E-mail: xiaoyu1qh1@163.com
### Paper : http://users.eecs.northwestern.edu/~xsh835/assets/cvpr2015_cascnn.pdf
****

train process
------
train process details in 
`process.txt`

test process
------
test process details in lp_test.py, you can run 
`python lp_test.py`  
    
you need to change some parameter as follows:  
- `caffe_root` : caffe root dir  
- `workspace`  : code dir  
- `img_dir`    : image dir  
- `img_list_file`: image list file  
- `min_lp_size`  : minimum license plate height size  
- `max_lp_size`  : maximum license plate height size  
- `save_res_dir` : save result dir  

run lp_test.py
- `load model`  
- `detect license plate`  
- `save results` 

For my dataset, I only use 12-net, 12-cal-net, 24-net and 48-cal-net.
I set up the ratio of w and h to 3:1. net input size is as follow:
- `12-net` : 12x4
- `12-cal` : 36x12
- `24-net` : 36x12
- `48-cal` : 72x48

You can change the parameters if you want.

More infomation, you can read the paper and see the code.

results
------



