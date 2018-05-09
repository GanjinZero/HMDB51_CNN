# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:26:50 2018

@author: ghm13
"""


import os 
import numpy as np  
import random
from sklearn.preprocessing import OneHotEncoder

path_save = 'C:/Users/ghm13/Desktop/inputdata'
dirs = os.listdir(path_save)
numbers = np.arange(0,51)
dictionary = {}
for i in range(len(dirs)):
    dictionary[dirs[i]] = numbers[i]
videos_train = []
videos_test = []
cls_train = []
cls_test = []
frame_num=5
def load_dataset():
    random.seed(310525)
    for motion in dirs:
        videos_path = os.path.join(path_save, motion)
        videos = os.listdir(videos_path)
        count = len(videos)
        # take a quarter of samples to be the testing_data, the others to be training data
        test_count = count//4
        test = random.sample(videos, test_count)
        train = list(set(videos).difference(set(test)))
        File = open("C:/Users/ghm13/Desktop/result/train_sample/" + motion + ".txt", "w") 
        for index in range(count-test_count):
            File.write(train[index] + "\n") 
        File.close()
        File = open("C:/Users/ghm13/Desktop/result/test_sample/" + motion + ".txt", "w") 
        for index in range(test_count):
            File.write(test[index] + "\n") 
        File.close()        
        for each_video in train:
            print (each_video)
            v = np.loadtxt(videos_path+'/'+each_video)
            v = v.reshape([frame_num, 60, 140, 3])
            videos_train.append(v)
            cls_train.append(dictionary[motion])
        for each_video in test:
            print (each_video)
            v = np.loadtxt(videos_path+'/'+each_video)
            v = v.reshape([frame_num, 60, 140, 3])
            videos_test.append(v)
            cls_test.append(dictionary[motion])
        onehot_encoder = OneHotEncoder(sparse=False)
    labels_train = onehot_encoder.fit_transform(cls_train.reshape(-1,1))
    labels_test = onehot_encoder.fit_transform(cls_test.reshape(-1,1))
    training_data = [videos_train, labels_train, cls_train] 
    testing_data = [videos_test, labels_test, cls_test]
    return training_data, testing_data



