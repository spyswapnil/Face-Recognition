import os
import cv2
import numpy as np
from PIL import Image

recog=cv2.createLBPHFaceRecognizer();
path='dataset'

def imgid(path):
    imgpath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imgpath in imgpath:
        faceimg=Image.open(imgpath).convert('L');
        facenp=np.array(faceimg,'uint8')
        ID=int(os.path.split(imgpath)[-1].split('.')[1])
        faces.append(facenp)
        ids.append(ID)
        print ID
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return ids,faces

ids,faces=imgid(path)
recog.train(faces,np.array(ids))
recog.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
