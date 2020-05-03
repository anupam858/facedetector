# Face Detector
Detecting face using opencv with input capabilities from ip cameras. Part of a bigger project.

Dependencies: 1. Numpy 2. OpenCV 3. imutils

imutils is a python library that has opencv convinience functions. It's used here to improve fps of ip cameras.


You can use an IP camera for live video input by providing the video link.

To stream video from your phone you can use droidcam. Download the app on your phone and pc to use it.
Usage instructions: https://www.dev47apps.com/droidcam/connect/

When using droidcam as input please close any existing videofeed that you recieve through it.
To get the video link, open the address of the feed in browser. In my case it was http://127.0.0.1:5050. Then right click on the video feed and copy the link to use it anywhere.


The face detection is done using the opencv dnn module which uses a caffe model as its brain.
