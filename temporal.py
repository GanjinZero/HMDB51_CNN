# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:32:03 2018

@author: ghm13
"""

import os
import cv2  
import numpy as np  
path = 'C:/Users/ghm13/Desktop/hmdb51_org'
dirs = os.listdir(path)
numbers = np.arange(1,52)
dictionary = {}
for i in range(len(dirs)):
    dictionary[dirs[i]] = numbers[i]
opticalflowset = []
classset = []

for motion in dirs: 
    videos_src_path = os.path.join(path, motion)
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




cap = cv2.VideoCapture("brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()