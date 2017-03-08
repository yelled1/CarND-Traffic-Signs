import tensorflow as tf
from tensorflow.contrib.layers import flatten

### Define your architecture here.
### Feel free to use as many code cells as needed.
#= Instead of 
def conv2d(x, W, bSz, strides=1, padding='SAME'):
    #conv2D wrapper, w/ biases & ReLu activation
    assert padding in ('SAME', 'VALID')
    b = tf.Variable(tf.zeros(bSz))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, padding='SAME'):
    return tf.nn.max_pool( x, ksize=[1,k,k,1], 
                            strides=[1,k,k,1],
                              padding=padding)

#28x28x1  1st layer shape the image to
def LeNet(x, dbg=False):
    # Hyperparameters
    mu = 0        # This assume that mean is Zero
    sigma = 0.1   # and sigma is .1
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    cnn1_Wgt = tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean=mu, stddev=sigma)) # 1 as input is 1
    cnn1_Layr = conv2d(x, cnn1_Wgt, 6, strides=1, padding='VALID')
    #conv2d(x, cnn1_Wgt, cnn1_b, strides=1, padding='SAME') #<tf.Tensor 'Relu_5:0' shape=(55000, 32, 32, 6)..
    if dbg: print( cnn1_Layr )
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.    
    pool1Layr = maxpool2d(cnn1_Layr, k=2, padding='SAME') # 'VALID shows same shape ???????
    if dbg: print( pool1Layr )
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    cnn2_Wgt = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma))
    cnn2_Layr = conv2d(pool1Layr, cnn2_Wgt, 16, strides=1, padding='VALID')
    if dbg: print( cnn2_Layr )
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2Layr = maxpool2d(cnn2_Layr, k=2, padding='VALID') # 'VALID o SAME has same shape ???????
    if dbg: print( pool2Layr )
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flattnd = flatten( pool2Layr )
    if dbg: print( flattnd )
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_Layr  = tf.matmul(flattnd, fc1_W) + tf.Variable(tf.zeros(120))
    if dbg: print( fc1_Layr )
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_Layr  = tf.nn.relu( tf.matmul(fc1_Layr, fc2_W) + tf.Variable(tf.zeros(84)) )
    if dbg: print( fc2_Layr )
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    logitL  = tf.matmul(fc2_Layr, fc3_W) + tf.Variable(tf.zeros(n_classes))
    if dbg: print( logitL )
    return logitL

"""
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#= below is the code to map sign classifier digit with image
#= Makes the data verification much much easier
import csv
SignD = {}
reader = csv.reader(open('signnames.csv', 'r'))
for row in reader: 
    try: SignD[int(row[0])] = row[1]
    except: print('Row not int', row)
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
"""
