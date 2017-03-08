import cv2, csv, glob, pickle
import numpy as np
import matplotlib.pyplot as plt

newD = {}
reader = csv.reader(open('./ExtSigns/labels.csv', 'r'))
for row in reader: 
    try: newD[row[0]] = row[1:]
    except: print('Row not int', row)
    #for fNm in glob.glob('./ExtSigns/*o.jpg'):

keys = list(newD.keys())
keys.sort()
X_ntest = np.zeros((len(keys), 32,32,3), np.int16)
y_ntest = np.zeros(len(keys), np.int16)
for i in range(len(keys)):
    fnm = './ExtSigns/'+keys[i]
    image = cv2.imread(fnm, flags=cv2.IMREAD_COLOR)
    X_ntest[i,:,:,:] = image
    y_ntest[i] = int( newD[keys[i]][0] )
    print(keys[i], newD[keys[i]])

X_ntest.shape
y_ntest

training_file = '../traffic-signs-data/train.p'
testing_file  = '../traffic-signs-data/test.p'

#with open(training_file, mode='rb') as f: train = pickle.load(f)
#X_train, y_train = train['features'], train['labels']
with open(testing_file, mode='rb') as f:  test = pickle.load(f)
X_test, y_test   = test['features'], test['labels']

#gray = cv2.cvtColor(X_test[0],cv2.COLOR_BGR2GRAY);gray.shape
#def cnv2gray(vec): return cv2.cvtColor(vec, cv2.COLOR_BGR2GRAY)
#xx = np.apply_over_axes(cv2.cvtColor, X_test, [0])

for i in range(y_ntest.shape[0]):
    ys = np.equal( y_test, y_ntest[i] )
    xs = X_test[ys, :, :, :]
    xf = xs.flatten().reshape(xs.shape[0], (32*32*3))
    print ( 'O %3d %3d %6.2f %6.2f' %(max(xf.flatten()),  min(xf.flatten()), \
                                    np.mean(xf.flatten()),np.std(xf.flatten())) )
    xn = X_ntest[i]
    print ( 'N %3d %3d %6.2f %6.2f' %(max(xn.flatten()),  min(xn.flatten()), \
                                    np.mean(xn.flatten()),np.std(xn.flatten())) )

