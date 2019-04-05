import cv2, numpy as np
import xlwrite,firebase.firebase_ini as fire;
import time
import sys
from playsound import playsound
start=time.time()
period=8
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('recognizer/face-trainner.yml');
flag = 0;
Id=0;
filename='filename';
dict = {
            'item1': 1
}
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.25, 5);
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
        Id,conf=recognizer.predict(roi_gray)
        print (Id)
        if(conf < 100):
           if(Id==1):
                Id='Vipasha Singh'
           if((str(Id)) not in dict):
                filename=xlwrite.output('attendance','class1',1,Id,'yes');
                dict[str(Id)]=str(Id);
                
           elif(Id==2):
                Id = 'Rohan'
           if ((str(Id)) not in dict):
               
               
               filename =xlwrite.output('attendance', 'class1', 2, Id, 'yes');
               dict[str(Id)] = str(Id);

           elif(Id==3):
               
               Id = 'Raveen'
           if ((str(Id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 3, Id, 'yes');
                dict[str(Id)] = str(Id);

           elif(Id==4):
               
               Id = 'Sonu'
           if ((str(Id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 4, Id, 'yes');
                dict[str(Id)] = str(Id);

        else:
             Id = 'Unknown, can not recognize'
             flag=flag+1
             break
        
        cv2.putText(img,str(Id)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    cv2.imshow('frame',img);
    #cv2.imshow('gray',gray);
    if flag == 10:
        playsound('transactionSound.mp3')
        print("Transaction Blocked")
        break;
    if time.time()>start+period:
        break;
    if cv2.waitKey(100) & 0xFF == ord('q'): 
        break;

cap.release();
cv2.destroyAllWindows();
