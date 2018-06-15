#face recognition in wild space
#
#Author: Xiaohong Zhao
#date: 2017.10.12

import numpy as np
import cv2
import libpysunergy  
import time
import random
import torch
from torch.autograd import Variable
import net_sphere
from feature import feature_comparer
import os


random.seed()

#read video from live cam
#cap = cv2.VideoCapture("rtsp://admin:xjtu123456@192.168.1.106/play1.sdp")
#cap = cv2.VideoCapture("rtsp://192.168.1.132:554/user=admin&password=&channel=1&stream=0.sdp?real_stream")

#read video from file


def res_conv(age, gender):
    if gender == 'male':
        gender = 'Male'
    else:
        gender = 'Female'
    age_up = random.randrange(0,10,1)
    age+=age_up
    if age <= 20:
        age_s = '<20'

    if age > 20 and age <= 25: 
        age_s = '20-25'

    if age > 25 and age <= 30: 
        age_s = '25-30'

    if age > 30 and age <= 35: 
        age_s = '30-35'

    if age > 35 and age <= 40: 
        age_s = '35-40'

    if age > 40 and age <= 45: 
        age_s = '40-45'

    if age > 45 and age <= 50: 
        age_s = '45-50'

    if age > 50 and age <= 55: 
        age_s = '50-55'

    if age > 55 and age <= 60: 
        age_s = '55-60'

    if age > 60:
        age_s = '>60'

    return age_s, gender


class face_analysis:
    def __init__(self,gpuid):
        self.frnet = net_sphere.sphere20a()
        self.frnet.load_state_dict(torch.load('model/sphere20a_20171020.pth'))
        self.frnet.cuda()
        self.frnet.eval()
        self.frnet.feature = True
        self.fcp = feature_comparer(512,0.8)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

        self.net, self.names = libpysunergy.load("data/face.data", "cfg/yolo_face_547.cfg", "weights/yolo_face_547.weights",gpuid)
        self.net2, self.names2 = libpysunergy.load("data/age1.1.data", "cfg/age1.1.cfg", "weights/age1.1.weights",gpuid)
        self.net3, self.names3 = libpysunergy.load("data/gender1.1.data", "cfg/gender1.1.cfg", "weights/gender1.1.weights",gpuid)

        self.top=1

    def run_frame(self,frame, fcp):
    	age_result = {}
    	gender_result = {}
        frame_original = frame.copy()
        (h, w, c) = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cfg_size = (608, 608)  # keep same as net input
        frame_input = cv2.resize(frame_rgb, cfg_size)
        threshold = 0.24
        dets = libpysunergy.detect(frame_input.data, w, h, c, threshold, self.net, self.names)

        for i in range(len(dets)):
            if dets[i][4] > 0 and (dets[i][5] - dets[i][4]) > 40:
                [fleft, fright, ftop, fbot] = dets[i][2:6]
                face_img = frame_original[ftop:fbot, fleft:fright].copy()
                (fh, fw, fc) = face_img.shape
                
                #face recognition
                face_image = cv2.resize(face_img, (112, 96))
                face_image =  face_image[:,:,::-1].transpose((2,0,1))
                face_image = (face_image[np.newaxis,:,:,:]-127.5)/128.0
                face_image = torch.from_numpy(face_image).float()
                face_image = Variable(face_image).cuda()
                
                output = self.frnet(face_image).data[0].tolist()
                ret, faceid = fcp.match(output)
                if ret:
                    age, gender = faceid.split(':')[0], faceid.split(':')[1]
                if not ret:
                #end of face recognition

                    dets2 = libpysunergy.predict(face_img.data, fw, fh, fc, self.top, self.net2, self.names2)
                    age = dets2[0][0]
                    dets3 = libpysunergy.predict(face_img.data, fw, fh, fc, self.top, self.net3, self.names3)
                    gender = dets3[0][0]
                    age, gender = res_conv(int(age), gender)
                    fcp.insert(output, str(age)+":"+str(gender))

                    if age not in age_result:
                        age_result[age] = 1
                    else:
                        age_result[age] += 1

                    if gender not in gender_result:
                        gender_result[gender] = 1
                    else:
                        gender_result[gender] += 1

        return age_result, gender_result
    #frame_skip: number to skip, default is 0 which means every frame will be processed
    def run_video(self,video_path,frame_skip=0):
        fcp = feature_comparer(512,0.8)
        age_result = {}
        gender_result = {}
        cap = cv2.VideoCapture(video_path)

        count = 0
        while 1:
            for i in range(0,frame_skip):
            	ret,frame = cap.read()
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            print count
            temp_age, temp_gender = self.run_frame(frame,fcp)

            for age in temp_age:
                if age not in age_result:
                    age_result[age] = temp_age[age]
                else:
                    age_result[age] += temp_age[age]

            for gender in temp_gender:
                if gender not in gender_result:
                    gender_result[gender] = temp_gender[gender]
                else:
                    gender_result[gender] += temp_gender[gender]            


        return age_result, gender_result
    def free(self):
        libpysunergy.free(self.net)
        libpysunergy.free(self.net2)
        libpysunergy.free(self.net3)
