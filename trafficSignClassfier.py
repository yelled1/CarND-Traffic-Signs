import pickle
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = '../traffic-signs-data/train.p'
testing_file  = '../traffic-signs-data/test.p'

with open(training_file, mode='rb') as f: train = pickle.load(f)
with open(testing_file, mode='rb') as f:  test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


n_train = X_train.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:]
n_classes = len(set(y_train))
print (n_classes)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import csv
SignD = {}
reader = csv.reader(open('signnames.csv', 'r'))
for row in reader: 
    try: SignD[int(row[0])] = row[1]
    except: print('Row not int', row)

import random
import numpy as np
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

print(y_train[index], SignD[y_train[index]])
plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
#plt.show()

"""
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation =\
    train_test_split(X_train, y_train, test_size=0.2, random_state=0)
print("Updated Image Shape: {}".format(X_train[0].shape))
"""
labelSet = set(train['labels'])
yLabelCnt = [ (ys, np.sum(np.equal(ys,train['labels']))) for ys in set(train['labels']) ]
print (max( yv[-1] for yv in yLabelCnt))
print (min( yv[-1] for yv in yLabelCnt))
plt.hist(list(yv[-1] for yv in yLabelCnt))
#plt.show()

import hashlib

print(train['features'].shape)

train_hashed = [ hashlib.sha1(x).digest() for x in train['features'] ]
test_hashed  = [ hashlib.sha1(x).digest() for x in test['features'] ]
print(len(train_hashed), len(set(train_hashed)))
print(len(test_hashed), len(set(test_hashed)))

test_in_train  = np.in1d(test_hashed,  train_hashed)
print('Cross Dups', np.sum(test_in_train))
test_in_train  = np.in1d(test_hashed,  train_hashed)
train_in_test  = np.in1d(train_hashed, test_hashed)
print('Cross Dups', np.sum(test_in_train))
#= we got No intrinsic for both train/test
#= & only 8 cross contamination, which is good
test_keep = ~test_in_train

#= Just repulling the data - ensures that code is clean
X_train, y_train = train['features'],     train['labels']
X_test,  y_test  = test['features'][test_keep], test['labels'][test_keep]
print(X_test.shape)

test_in_train.shape
te_ix = list( np.where(test_in_train)[0] ); print(te_ix) 
tr_ix = list( np.where(train_in_test)[0] ); print(tr_ix)
 
i=0
train['features'][tr_ix[i]].shape
image_train = train['features'][tr_ix[i]].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image_train,  cmap="gray")

image_test = test['features'][te_ix[i]].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image_test,  cmap="gray")
plt.show()

if 0:
#for i in range(len(te_ix))[1:]:
    print(train['labels'][tr_ix[i]], test['labels'][te_ix[i]])
    Left = train['features'][tr_ix[i]].squeeze()
    Right = test['features'][te_ix[i]].squeeze()

    f, axarr = plt.subplots(1, 2)
    axarr[0, 0].set_title('Train')
    axarr[0, 0].imshow(Left, cmap='gray')
    axarr[0, 0].set_title('Test')
    axarr[0, 0].imshow(Right, cmap='gray')
    plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
    plt.show()
    plt.close('all')
