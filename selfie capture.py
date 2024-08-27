import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
face_harr = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    _,frame = cap.read()
    
    face_cas = face_harr.detectMultiScale(frame,1.1,4)
    for x,y,w,h in face_cas:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        
    cv.imshow('frame',frame)
    if cv.waitKey(5) &0xff==ord('q'):
        cv.imwrite('myself.jpg',frame)
        break
cap.release()
cv.destroyAllWindows()
