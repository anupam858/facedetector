import numpy as np
import cv2
import os
from imutils.video import WebcamVideoStream
import time
# set default current directory and load caffe model
path = os.path.dirname(__file__)
prototxt_path = os.path.join(path + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(path + 'model_data/weights.caffemodel')

net = cv2.dnn.readNetFromCaffe(prototxt_path,caffemodel_path)

# Start video 

print("Use default webcam? \n y/n : ")
cam= 0

if(input()=='n'):
    
    print("Enter video stream output link: ")
    cam = input()

video = WebcamVideoStream(cam).start()
time.sleep(1)

while(True):

    frame = video.read()
    frame = cv2.flip(frame,flipCode=1)
    ret=True
    if ret==True:
        
        (h, w) = frame.shape[:2]
        
        # Convert image to blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)),1.0, (300,300), (104.0,177.0,123.0))
        net.setInput(blob)
        
        # detection
        detection = net.forward()
        
        # iterate through all the recognized faces 
        
        for i in range(0, detection.shape[2]):
            
            confidence = detection[0,0,i,2]
            
            # continue only if confidence level is greater than 50% 
            if (confidence<0.5):
                continue
                
            box  = detection[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    #frame = np.dstack([frame, frame, frame])
    cv2.imshow("Video Feed",frame)
    
    key = cv2.waitKey(2)
    
    if(key==ord('q')):
        break
        
cv2.destroyAllWindows()
video.stop()