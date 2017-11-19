import cv2
import dlib
from imutils import face_utils
import imutils
import numpy as np
from numpy.linalg import norm

def bb_to_rect(bb):
    print bb
    x= dlib.rectangle(int(bb[0]), int(bb[1]), int(bb[0] + bb[2]), int(bb[0] + bb[3]))
    print x
    return x

def detectFace(img, option = 1):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = []
    if option == 1:
        detector = dlib.get_frontal_face_detector()
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            print type(shape)
            faces.append((face_utils.rect_to_bb(rect), shape))
    elif option == 2:
        detector = cv2.CascadeClassifier('/home/turja/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
        face_roi = detector.detectMultiScale(gray, 1.3, 5)
        for face in face_roi:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            rect = bb_to_rect(face)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            faces.append((face, shape))

    return faces

def detectLipMovement(mouth, scale = 1):
    # take 3 points from the lips
    point_1 = np.array([mouth[6][0], mouth[6][1]])
    point_2 = np.array([mouth[3][0], mouth[3][1]])
    point_3 = np.array([mouth[9][0], mouth[9][1]])

    # create two vectors
    v1= point_2 - point_1
    v2 = point_3 - point_1

    #find angles between them
    c = np.dot(v1, v2)/norm(v1)/norm(v2)
    return np.arccos(np.clip(c, -1, 1))/scale

def detectLipMovement2(mouth, scale = 1):
    pivots = [mouth[13], mouth[14], mouth[15]]
    target = [mouth[19], mouth[18], mouth[17]]

    dist = 0
    for a, b in zip(pivots, target):
        dist = dist + norm(a - b)
    return dist/(len(mouth))/scale

def detectLipMovement3(mouth, scale = 1):
    list1 = np.concatenate((mouth[0:6], mouth[12:16]), axis=0)
    print len(list1)
    list2 = np.concatenate((mouth[7:12], mouth[16:]))
    print len(list2)
    dist = 0
    for m1 in list1:
        for m2 in list2:
            dist =  dist + norm(m1 - m2)

    return dist/len(list1)/len(list2)/scale

def find_angle(v1, v2):
    print('Vect1', v1)
    print('Vect2', v2)
    print v2
    c = np.dot(v1, v2) / norm(v1) / norm(v2)
    angle = np.arccos(np.clip(c, -1, 1))
    if angle <= 180:
        return angle
    else:
        return 360 - angle

def makeLine(p1, p2):
    return np.array(p1-p2)
    #return np.array(p2[0], p2[1]) - np.array(p1[0], p1[1])

def find_mid(p, q):
    return ((p[0] + q[0])/2, (p[1] + q[1])/2)
def featureSet(points, scale):
    pivot = find_mid(points[0], points[6])

    angle_feature_set = []
    for p in points:
        angle_feature_set.append(find_angle_from_points(points[6], pivot, p))
        angle_feature_set.append(norm(p - pivot)/scale)

    return angle_feature_set

def find_angle_from_points(p1, p2, p3):
    print 'Finding angle from 3 points'
    v1 = makeLine(p2, p1)
    v2 = makeLine(p2, p3)
    return find_angle(v1, v2)

def featureSetUsingHull(points, scale):
    hull = cv2.convexHull(points)
    feat = []
    hull_len = len(hull)
    #adding distance in feature set
    for i in range(0, len(hull)):
        p1 = hull[i]
        p2 = hull[(i + 1)%hull_len]
        feat.append(norm(p1[0] - p2[0])/scale)


    #adding angles in feature set
    for i in range(0, len(hull)):
        p1 = hull[i]
        p2 = hull[(i + 1) % hull_len]
        p3 = hull[(i + 2) % hull_len]

        print p1[0], p2[0], p3[0]
        feat.append(find_angle_from_points(p1[0], p2[0], p3[0]))

    print 'Lenn: :',len(feat)
    return feat

def featureSetUsingOnlyImprotantPoints(points):
    pIndex = [0, 2, 3, 4, 6, 8, 10]
    feat = []

    for i in range(0, len(pIndex)):
        feat.append(find_angle_from_points(points[pIndex[i]], points[pIndex[(i + 1)%len(pIndex)]], points[pIndex[(i + 2)%len(pIndex)]]))

    feat.append(norm(points[0] - points[6]) * 1.0 / norm(points[3] - points[9]))
    return feat

def find_nearest_neighbor(val, seedList):
    dist = -1
    nseed = None
    count = -1
    index = 0
    for seed in seedList:
        count = count + 1
        p = norm(np.array(val) - np.array(seed))
        print p
        if dist == -1 or p < dist:
            dist = p
            nseed = seed
            index = count

    return nseed, index



