# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import cv2  
path = 'C:/Users/ghm13/Desktop/hmdb51_org'
path_cap = 'C:/Users/ghm13/Desktop/Capture'
dirs = os.listdir(path)

for motion in dirs:
    os.mkdir(path_cap + '/' + motion) 
    videos_src_path = os.path.join(path, motion)
    videos_save_path = os.path.join(path_cap, motion)
    videos = os.listdir(videos_src_path)
    videos = filter(lambda x: x.endswith('avi'), videos)
    for each_video in videos:
        print (each_video)
        # get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')
        os.mkdir(videos_save_path + '/' + each_video_name)               
        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'
        # get the full path of each video, which will open the video to extract frames
        each_video_full_path = os.path.join(videos_src_path, each_video)
        cap = cv2.VideoCapture(each_video_full_path)
        frame_count = 0
        success = True 
        while(success):
            success, frame = cap.read()
            frame_count = frame_count + 1
            
        frame_num=5 
        #frame_num indicates the number of frames captured in one video.
        success = True 
        count = 0
        i = 1
        cap = cv2.VideoCapture(each_video_full_path)
        while (success):
            success, frame = cap.read()
            count = count + 1
            params = []
            if (count==frame_count//frame_num*(i-1)+1):
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % i, frame, params)
                i = i+1
            if (i > frame_num):
                break
    cap.release()
 

os.chdir('C:/Users/ghm13/Desktop/Capture/brush_hair')
os.getcwd()
import imageio
img = imageio.imread('April_09_brush_hair_u_nm_np1_ba_goo_0\April_09_brush_hair_u_nm_np1_ba_goo_0_48.jpg')
print(img.shape) 

 
