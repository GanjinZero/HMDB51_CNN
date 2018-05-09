# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import cv2  
import numpy as np  
path = 'C:/Users/ghm13/Desktop/hmdb51_org'
path_save = 'C:/Users/ghm13/Desktop/inputdata'
dirs = os.listdir(path)
numbers = np.arange(1,52)
dictionary = {}
for i in range(len(dirs)):
    dictionary[dirs[i]] = numbers[i]


frame_num=5 
#frame_num indicates the number of frames we want to capture in one video.
for motion in dirs:
    os.mkdir(path_save + '/' + motion) 
    videos_src_path = os.path.join(path, motion)
    videos_save_path = os.path.join(path_save, motion) 
    videos = os.listdir(videos_src_path)
    videos = filter(lambda x: x.endswith('avi'), videos)
    for each_video in videos:
        print (each_video)
        # get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')              
        # get the full path of each video, which will open the video to extract frames
        each_video_full_path = os.path.join(videos_src_path, each_video)
        cap = cv2.VideoCapture(each_video_full_path)
        frame_count = 0
        success = True 
        while(success):
            success, frame = cap.read()
            frame_count = frame_count + 1     
        success = True 
        count = 0
        i = 1
        cap = cv2.VideoCapture(each_video_full_path)
        imagesset = []
        while (success):
            success, frame = cap.read()
            count = count + 1
            params = []
            if (count==frame_count//frame_num*(i-1)+1):
                width = np.array(frame).shape[1]
                # normalize the image size to 240*560*3
                if (width < 560):
                    d = int((560 - width)/2)
                    frame0 = np.zeros(240*d*3, dtype = 'i').reshape([240, d, 3])
                    frame = np.concatenate((frame0, frame), axis = 1)
                    frame = np.concatenate((frame, frame0), axis = 1)
                # 4*4 max-pooling
                frame1 = np.zeros(60*140*3, dtype = 'i').reshape([60,140,3])
                for j in range(60):
                    for k in range(140):
                        for l in range(3):
                            frame1[j,k,l]=max(frame[(4*j+1):(4*j+4),(4*k+1):(4*k+4),l].flatten())
                imagesset.append(frame1)
                i = i + 1
            if (i > frame_num):
                break
        np.savetxt(videos_save_path + "/" + each_video_name + ".txt" , np.array(imagesset).flatten(), fmt="%d")
    cap.release()


    






