# TryLip
The aim of this project is to detect various types of expressions such as Smiling, Laughing, Pouting, Opening mouth etc using human lip movement in real time.
The main challenge of detecting any kind of lip movement is to find some landmark points on the lips. To do this, I first 
localized the face in the frame and then used the facial landmark detector of the dlib library -- an implementation of this 
work -- to detect the lip landmarks on the face. From these points, I crafted some features and then used k-means clustering 
algorithm to partition between five types of lip states. The clusters being learned, I manually assigned some labels to them and thus completing this project. The end result is a lip movement detector that can detect five types of lip movements in real-time.
The detector is robust to varying distances from the camera. Here is a demonstration of this project.

![No Movement](No_movement.png?raw=true "No Movement") 
![Smiling](Smiling.png?raw=true "Smiling")
![Laughing](Lauging.png?raw=true "Laughing")
![Pouting](Pouting.png?raw=true "Pouting")
![Opening](Open.png?raw=true "Opening")

# Face Localization:
To localize face in a frame, I used dlibs frontal face detector.


# Detecting Landmarks:
To detect the landmarks on the lips, I used dlibs landmark detection library. There are 20 such points (from index 49 to 68). The indices of the points are given in the following figure.

![Landmark](Landmarks.png?raw=true "Visualizing each of the 68 facial coordinate points from the iBUG 300-W dataset ")

# Extracting Features From Landmarks:
To extract features from landmarks, I have done the following two steps:
Take the points with index 49,  51, 52, 53, 55, 57, 59.
Create a polygon using these points and take the interior angles of the polygon as feature.

# Unsupervised Learning model:
Now that I have some features to work with, I used k-means clustering to differentiate between the shapes of the polygon. To train the model, I just recorded a video of me demonstrating different shapes. The model was able to learn the clusters just by watching one minute of video which is fascinating.

# Future Work:
I intend to extend this work to detect various shapes of mouth when human talks. My aim is to interpret what a person is saying just by analyzing the lip movement. This can help mutes to talk about using vocal chords.
