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
    train process details in process.txt

test process
------
test process details in lp_test.py, you can run `python lp_test.py`  
you need to change some parameter as follows:  
`caffe_root` : caffe root dir  
`workspace`  : code dir  
`img_dir`    : image dir  
`img_list_file`: image list file  
`min_lp_size`  : minimum license plate height size  
`max_lp_size`  : maximum license plate height size  
`save_res_dir` : save result dir  

results
------



