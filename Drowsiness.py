import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import winsound
frequency=2500
duration=1000
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
model=tf.keras.models.load_model('drowsiness_model.h5')
capture=cv.VideoCapture(0)
count=0
flag=0
left_eye_pred=[]
right_eye_pred=[]
while(True):
    isTrue,frame=capture.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    haar_cascade_face=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rect=haar_cascade_face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(25,25))
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    haar_cascade_left_eye=cv.CascadeClassifier('haarcascade_left_eye.xml')
    haar_cascade_right_eye=cv.CascadeClassifier('haarcascade_right_eye.xml')
    faces_rect_left_eye=haar_cascade_left_eye.detectMultiScale(gray)
    for (x,y,w,h) in faces_rect_left_eye:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=8)
    faces_rect_right_eye=haar_cascade_right_eye.detectMultiScale(gray)
    for (x,y,w,h) in faces_rect_right_eye:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=8)
    if(len(faces_rect_left_eye)!=0):
        x=faces_rect_left_eye[0][0]
        y=faces_rect_left_eye[0][1]
        w=faces_rect_left_eye[0][2]
        h=faces_rect_left_eye[0][3]
       # crp=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        crp=frame[y:y+h,x:x+w]
        crp=cv2.cvtColor(crp,cv2.COLOR_BGR2GRAY)
        img_prc = cv.resize(crp,(24,24))
        img_prc=img_prc/255
        img_prc=img_prc.reshape(24,24,-1)
        x = np.expand_dims(img_prc, axis=0)
        #x = preprocess_input(x, mode='caffe')
        left_eye_pred=model.predict_classes(x)
        print(left_eye_pred[0])
    if(len(faces_rect_right_eye)!=0):
        x=faces_rect_right_eye[0][0]
        y=faces_rect_right_eye[0][1]
        w=faces_rect_right_eye[0][2]
        h=faces_rect_right_eye[0][3]
        crp=frame[y:y+h,x:x+w]
        crp=cv2.cvtColor(crp,cv2.COLOR_BGR2GRAY)
        img_prc = cv.resize(crp,(24,24))
        img_prc=img_prc/255
        img_prc=img_prc.reshape(24,24,-1)
        x = np.expand_dims(img_prc, axis=0)
        right_eye_pred=model.predict_classes(x)
        print(right_eye_pred[0])
        if(left_eye_pred[0]==1 or left_eye_pred[0]==1):
            count=0
        else:
            count=count+1
        
            
        #print(count)
        if count>3:
            count=0
            winsound.Beep(frequency,duration)
            flag=flag+1
            cv.putText(frame,'Drowsiness Alert!!!!!!',(225,225),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,255),2)
        if flag>2:
            winsound.Beep(5500,3000)
            
            cv.putText(frame,'We recommend you stop',(35,35),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,255),2)
            
            

            

    
#     cv.putText(frame,'Drowsiness Alert!!!!!!',(225,225),cv.FONT_HERSHEY_TRIPLEX,1.0,(255,255,255),2)    
    cv.imshow('frame',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    
    
capture.release()
cv.destroyAllWindows()   