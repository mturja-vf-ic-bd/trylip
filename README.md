# trylip
The aim of this project is to detect various types of expressions such as Smiling, Laughing, Pouting, Opening mouth etc using human lip movement in real time.
The main challenge of detecting any kind of lip movement is to find some landmark points on the lips. To do this, I first 
localized the face in the frame and then used the facial landmark detector of the dlib library -- an implementation of this 
work -- to detect the lip landmarks on the face. From these points, I crafted some features and then used k-means clustering 
algorithm to partition between five types of lip states. The clusters being learned, I manually assigned some labels to them and thus completing this project. The end result is a lip movement detector that can detect five types of lip movements in real-time.
The detector is robust to varying distances from the camera. Here is a demonstration of this project.
