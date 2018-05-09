# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:54:36 2018

@author: GanJinZERO
"""

import numpy as np

classset_test = np.zeros([1673,51])
for j in range(1,1674):
    classset_test[j-1][cls_test[j-1]]=1
classset_train = np.zeros([5093,51])
for j in range(1,5094):
    classset_train[j-1][cls_train[j-1]]=1

x=videos_test[0]
xx=np.transpose(x,[1,2,0,3])
xxx=[xx.reshape([60,140,15])]
imageset_test=xxx
for j in range(2,1674):
    x=videos_test[j-1]
    xx=np.transpose(x,[1,2,0,3])
    xxx=xx.reshape([60,140,15])
    imageset_test=np.concatenate((imageset_test,[xxx]))
#imageset_test.reshape([1673,60,140,3])

x=videos_test[0]
xx=np.transpose(x,[1,2,0,3])
xxx=[xx.reshape([60,140,15])]
imageset_train=xxx
for j in range(2,5094):
    x=videos_train[j-1]
    xx=np.transpose(x,[1,2,0,3])
    xxx=xx.reshape([60,140,15])
    imageset_train=np.concatenate((imageset_train,[xxx]))
#imageset_test.reshape([5093,60,140,3])