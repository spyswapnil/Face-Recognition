import cv2
import numpy as np

facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
rec=cv2.createLBPHFaceRecognizer();
rec.load('recognizer\\trainingData.yml')
Id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,2,3,3,4)
while(True):
    ret,pic=cam.read()
    gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    faces= facedetect.detectMultiScale(gray,2,10)
    for(x,y,w,h) in faces:
        cv2.rectangle(pic,(x,y),(x+w,y+h),(0,0,255),4)
        Id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(Id==1):
            Id='Swapnil'
        elif(Id==2):
            Id='Guru'
        elif(Id==6):
            Id='Yogi'
        elif(Id==7):
            Id='Ayush'
        elif(Id==8):
            Id='Pulkit'
        elif(Id==9):
            Id='Maa'
        cv2.cv.PutText(cv2.cv.fromarray(pic),str(Id),(x,y+h),font,255)
    cv2.imshow('frame',pic)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
