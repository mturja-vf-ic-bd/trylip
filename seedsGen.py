import os
import tools
import cv2
from numpy.linalg import norm
import pickle
import glob
from matplotlib import pyplot as plt

frame = cv2.imread('Seeds/seed1.jpg')
print frame
path = 'Seeds/'
fl = open('seed', 'w+')
seedfeat = []

for i in range(1,6):
    pathName = path + 'seed' + str(i) + '.jpg'
    print pathName
    frame = cv2.imread(pathName)
    roi_faces = tools.detectFace(frame)
    for face, landmarks in roi_faces:
        print 'I am here'
        x, y, w, h = face
        i = 48
        j = 68
        val = tools.detectLipMovement2(landmarks[i: j], scale=norm(landmarks[0] - landmarks[16]))
        afs = tools.featureSet(landmarks[i: j], scale=norm(landmarks[0] - landmarks[16]))
        print len(afs)
        seedfeat.append(afs)




pickle.dump(seedfeat, fl)
fl.close()
print seedfeat
with open('seed', 'rb') as fl:
    print pickle.load(fl)