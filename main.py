import numpy as np
import tools
import cv2
from matplotlib import pyplot as plt
import time
import datetime
from numpy.linalg import norm
import sounddevice as sd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pickle


cap = cv2.VideoCapture(0)
graph_data_set = []
stage = 0
'''
#setting up sound recording environment
sd.default.samplerate = 44100
fs = 44100
duration = 10
sound = sd.rec(int(duration * fs), samplerate=fs, channels=1)'''

featureSet = []
kmeans = joblib.load('kmeansModel.pkl')
seeds = pickle.load(open('seed', 'rb'))
frameLabel = 0
featureSet = []
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frameLabel = frameLabel + 1

    if ret == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 30, (640,480))
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_faces = tools.detectFace(frame, 1)
        for face, landmarks in roi_faces:
            x, y, w, h = face
            i = 48
            j = 68
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # loop over the subset of facial landmarks, drawing the
            # specific face part
            '''
            (p, q) = landmarks[48]
            (s, t) = landmarks[54]
            (p, q) = ((p + s)/2, (q + t)/2)
            for (x, y) in landmarks[i:j]:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.line(frame, (x, y), (p, q), (0, 255, 0), 1)'''

            npl = landmarks[i:j]
            pIndex = [0, 2, 3, 4, 6, 8, 10]

            for a in range(0, len(pIndex)):
                (p1, q1) = npl[pIndex[a]]
                (p2, q2) = npl[pIndex[(a+1)%len(pIndex)]]
                cv2.circle(frame, (p1, q1) , 1, (0, 0, 255), -1)
                cv2.line(frame, (p1, q1) , (p2, q2), (0, 255, 0), 1)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            #val = tools.detectLipMovement2(landmarks[i : j], scale = norm(landmarks[0] - landmarks[16]))
            afs = tools.featureSetUsingOnlyImprotantPoints(landmarks[i : j])
            featureSet.append(afs)
            temp = []
            temp.append(afs)
            featureSet.append(afs)
            temp = np.array(temp)
            where_are_NaNs = np.isnan(temp)
            temp[where_are_NaNs] = 0
            #val1 = 3
            val1 = kmeans.predict(np.array(temp))
            #minseed, val1 = tools.find_nearest_neighbor(afs, seeds)
            print('kmean prediction: ', val1)
            #graph_data_set.append((val, st))
            writeText = None
            if val1 == 2:
                writeText = 'Smiling'
            elif val1 == 1:
                writeText = 'Laughing'
            elif val1 == 0:
                writeText = 'Pouting'
            elif val1 == 3:
                writeText = 'No movement'
            elif val1 == 4:
                writeText = 'Open'

            cv2.putText(frame, writeText, (x + 40, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Display the resulting frame
        cv2.imshow('frame',frame)
        cv2.imwrite('output.jpg', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
i = 0
'''smooth_data = []
graph_data, time = zip(*graph_data_set)
for i in range(1, len(graph_data) - 1):
    smooth_data.append((graph_data[i - 1] + 3*graph_data[i] + graph_data[i + 1])/5)'''

'''smooth_dif = []
th = 0.7
for i in range(1, len(diff) - 1):
    smooth_dif.append((diff[i - 1] + 3 * diff[i] + diff[i + 1])/5)
    if i > 0 and smooth_dif[i] - smooth_dif[i - 1] > th:'''

# cancel noise in sound
'''for i in range(2, len(sound)-2):
    a = 0
    for j in range(-2,3):
        a = a + sound[i + j]
    sound[i] = a/5'''

#show response graph
#plt.plot(graph_data)
'''
plt.subplot(2,1,1)
plt.plot(sound)
plt.title('Actual Sound')
plt.subplot(2,1,2)
plt.plot(smooth_data)
plt.title('Estimation from lips')
plt.show()

plt.plot(graph_data)
plt.show()
'''
#Kmeans
#print featu.shape
'''
print len(featureSet)
with open('history_three', 'a+b') as f:
    pickle.dump(featureSet, f)



with open('history_three', 'rb') as f:
    while 1:
        try:
            tmpfeat = pickle.load(f)
            for elem in tmpfeat:
                featureSet.append(elem)
        except EOFError:
            break

print type(featureSet)
featureSet = np.array(featureSet)
where_are_NaNs = np.isnan(featureSet)
featureSet[where_are_NaNs] = 0
#seeds = pickle.load(open('seed', 'rb'))
#seeds = np.array(seeds)
#print seeds.shape
kmeans = KMeans(n_clusters=5, random_state=0).fit(featureSet)
#joblib.dump(kmeans, 'kmeansModel.pkl')'''


