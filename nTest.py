import cv2, csv, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lenetModel import *
import hashlib
import pickle

training_file = '../traffic-signs-data/train.p'
testing_file  = '../traffic-signs-data/test.p'
with open(training_file, mode='rb') as f: train = pickle.load(f)
with open(testing_file, mode='rb') as f:  test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test   = test['features'], test['labels']

train_hashed = [ hashlib.sha1(x).digest() for x in train['features'] ]
test_hashed  = [ hashlib.sha1(x).digest() for x in test['features'] ]
print(len(train_hashed), len(set(train_hashed)))
print(len(test_hashed), len(set(test_hashed)))
test_in_train  = np.in1d(test_hashed,  train_hashed)
train_in_test  = np.in1d(train_hashed, test_hashed)
print('Cross Dups', np.sum(test_in_train))
test_keep = ~test_in_train
X_train, y_train = train['features'],     train['labels']
X_test,  y_test  = test['features'][test_keep], test['labels'][test_keep]


test_in_train.shape
te_ix = list( np.where(test_in_train)[0] ); print(te_ix) 
tr_ix = list( np.where(train_in_test)[0] ); print(tr_ix) 
train['features'][tr_ix[0]].shape
for i in range(len(te_ix)):
    print(train['labels'][tr_ix[i]], test['labels'][te_ix[i]])

n_train = X_train.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#= below is the code to map sign classifier digit with image
#= Makes the data verification much much easier
SignD = {}
reader = csv.reader(open('signnames.csv', 'r'))
for row in reader: 
    try: SignD[int(row[0])] = row[1]
    except: print('Row not int', row)
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)


EPOCHS = 30
BATCH_SIZE = 128
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# ## Model Evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ## Train the Model
# Run the training data through the training pipeline to train the model.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    print('BATCH_SIZE', BATCH_SIZE)
    saver.save(sess, 'lenet')
    print("Model saved")


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
    plt.figure(figsize=(1,1))
    plt.imshow(image.squeeze(), cmap="gray")

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    ntest_accuracy = evaluate(X_ntest, y_ntest)
    print("New Test Accuracy = {:.3f}".format(ntest_accuracy))
