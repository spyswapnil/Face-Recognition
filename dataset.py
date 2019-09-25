import cv2
import numpy as np

facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)

Id=raw_input('enter user id=')
samplenum=0;
while(True):
    ret,pic=cam.read()
    gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    faces= facedetect.detectMultiScale(gray,2,10)
    for(x,y,w,h) in faces:
        samplenum=samplenum+1;
        cv2.imwrite("dataset/user."+str(Id)+"."+str(samplenum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(pic,(x,y),(x+w,y+h),(0,0,255),4)
        cv2.waitKey(200);
    cv2.imshow('frame',pic)
    cv2.waitKey(1);
    if(samplenum>100):
        break

cam.release()
cv2.destroyAllWindows()
