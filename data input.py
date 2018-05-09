# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:35:45 2018

@author: GanJinZERO
"""

#cls_test cls_train
#videos_test videos_train
#classset_test imageset_test
#classset_train imageset_train

import numpy as np

classset_test = np.zeros([1673,51])
for j in range(1,1674):
    classset_test[j-1][cls_test[j-1]]=1
classset_train = np.zeros([5093,51])
for j in range(1,5094):
    classset_train[j-1][cls_train[j-1]]=1
    
imageset_test1=videos_test[0]
imageset_test=[imageset_test1[3]]
for j in range(2,1674):
    x=videos_test[j-1]
    xx=[x[2]]
    imageset_test=np.concatenate((imageset_test,xx))
#imageset_test.reshape([1673,60,140,3])

imageset_train1=videos_train[0]
imageset_train=[imageset_train1[3]]
for j in range(2,5094):
    x=videos_train[j-1]
    xx=[x[2]]
    imageset_train=np.concatenate((imageset_train,xx))
#imageset_test.reshape([5093,60,140,3])