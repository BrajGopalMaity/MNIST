import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

#define MACROS
#learning rate
LRATE = 1e-4

#training iterations
N = 25000  

#dropout rate used to prevent overfitting          
DRATE = 0.5

#batchsize to train in batches
BSIZE = 50

#no of images taken in crossvalidation
VSIZE = 1000

#read csv data from the file and convert into a matrix
data = pd.read_csv('/home/bgm/Desktop/Big Data/train4.csv')

#print to check the read data
print(data.shape)

#convert each of the rows of data into a single image of 784 dimensions/features i.e a 28x28 image with 784 pixels from 1st column to 785 column
images = data.iloc[:,1:].values
images = images.astype(np.float)

# normalize the data to have value from 0.0 to 1.0
images = np.multiply(images, 1.0 / 255.0)

imageSize = images.shape[1]

#each image will have a width and height 28 or sqrt(784)
imageWidth = imageHeight = np.ceil(np.sqrt(imageSize)).astype(np.uint8)
print ('imageWidth => {0}\nimageHeight => {1}'.format(imageWidth,imageHeight))

# The corresponding labels are numbers between 0 and 9, describing which digit a given image is of.
labels = data[[0]].values.ravel()

#no. of labels
lCount = np.unique(labels).shape[0]

#Convert class labels from scalars to one hot vectors
def scalarToVector(labelScalar, classes):
    numLabels = labelScalar.shape[0]
    indexOffset = np.arange(numLabels) * classes
    labelVector = np.zeros((numLabels, classes))
    labelVector.flat[indexOffset + labelScalar.ravel()] = 1
    return labelVector

labels = scalarToVector(labels, lCount)
labels = labels.astype(np.uint8)

# split data into training & validation
validationImages = images[:VSIZE]
validationLabels = labels[:VSIZE]

trainImages = images[VSIZE:]
trainLabels = labels[VSIZE:]

# Neural Network

# weight initialization
def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#leaky Relu
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

#pooling
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# images
x = tf.placeholder('float', shape=[None, imageSize])
# labels
y1 = tf.placeholder('float', shape=[None, lCount])

# first convolutional layer
w1 = weight([5, 5, 1, 32])
b1 = bias([32])

# Convert the 2d matrix into a 4d tensor to be fed as input to the convolution layer as one of its arguments
image = tf.reshape(x, [-1,imageWidth , imageHeight,1])

#output of the first convolution layer through relu
#h1 = lrelu(conv2d(image, w1) + b1)
h1 = tf.nn.relu(conv2d(image, w1) + b1)
#after pooling to reduce dimensions to 14x14
hp1 = max_pool(h1)

# second convolutional layer
w2 = weight([5, 5, 32, 64])
b2 = bias([64])

#h2 = lrelu(conv2d(hp1, w2) + b2)
h2 = tf.nn.relu(conv2d(hp1, w2) + b2)
hp2 = max_pool(h2)

# third 1x1 convolution layer
w2_2 = weight([1,1,64,64])
b2_2 = bias([64])
h2_2 = tf.nn.relu(conv2d(hp2, w2_2)+b2_2)

# fully connected layer for the second layer where imagesize is reduced to 7x7 dimension after pooling
w3 = weight([7 * 7 * 64, 1024])
b3 = bias([1024])

hp2_flat = tf.reshape(h2_2, [-1, 7*7*64])

h3 = tf.nn.relu(tf.matmul(hp2_flat, w3) + b3)

# dropout to prevent overfitting
drop = tf.placeholder('float')
h3_drop = tf.nn.dropout(h3, drop)

# readout layer for deep net
w4 = weight([1024, lCount])
b4 = bias([lCount])

#softmax regression expression
y = tf.nn.softmax(tf.matmul(h3_drop, w4) + b4)

#evaluate model using cross-entropy and minimize the cross-entropy using ADAM optimizer
# cost function
cEntropy = -tf.reduce_sum(y1*tf.log(y))

# optimisation function
steps = tf.train.AdamOptimizer(LRATE).minimize(cEntropy)

# evaluation
actualPredict = tf.equal(tf.argmax(y,1), tf.argmax(y1,1))

accuracy = tf.reduce_mean(tf.cast(actualPredict, 'float'))

# prediction function
predict = tf.argmax(y,1)

#training data in batches
epochsOver = 0
epochIndex = 0
examples = trainImages.shape[0]

# serve data by batches
def getBatch(bSize):
    
    global trainImages
    global trainLabels
    global epochIndex
    global epochsOver
    
    start = epochIndex
    epochIndex += bSize
       
    if epochIndex > examples:
        # finished epoch
        epochsOver += 1
        # shuffle the data
        perm = np.arange(examples)
        np.random.shuffle(perm)
        trainImages = trainImages[perm]
        trainLabels = trainLabels[perm]
        # start next epoch
        start = 0
        epochIndex = bSize
        assert bSize <= examples
    end = epochIndex
    return trainImages[start:end], trainLabels[start:end]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
trainAccuracies = []
validationAccuracies = []
xRange = []

dStep=1

for i in range(N):

    #get new batch
    batchX, batchY = getBatch(BSIZE)        
    
    #check accuracy
    if i%dStep == 0 or (i+1) == N:
        
        trainAccuracy = accuracy.eval(feed_dict={x:batchX, y1: batchY, drop: 1.0})       
        if(VSIZE):
            validationAccuracy = accuracy.eval(feed_dict={ x: validationImages[0:BSIZE], y1: validationLabels[0:BSIZE], drop: 1.0})                                  
            print('trainingAccuracy / validationAccuracy => %.2f / %.2f for step %d'%(trainAccuracy, validationAccuracy, i))
            
            validationAccuracies.append(validationAccuracy)
            
        else:
             print('trainingAccuracy => %.4f for step %d'%(trainAccuracy, i))
        trainAccuracies.append(trainAccuracy)
        xRange.append(i)
        
        # increase display step
        if i%(dStep*10) == 0 and i:
            dStep *= 10
    # train on batch
    sess.run(steps, feed_dict={x: batchX, y1: batchY, drop: DRATE})

# check final accuracy on validation set  
if(VSIZE):
    validationAccuracy = accuracy.eval(feed_dict={x: validationImages, y1: validationLabels, drop: 1.0})
    print('validationAccuracy => %.4f'%validationAccuracy)
    plt.plot(xRange, trainAccuracies,'-b', label='Training')
    plt.plot(xRange, validationAccuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()

# read test data from CSV file 
test_images = pd.read_csv('/home/bgm/Desktop/Big Data/test.csv').values
test_images = test_images.astype(np.float)

# normalize the data to have value from 0.0 to 1.0
test_images = np.multiply(test_images, 1.0 / 255.0)

#prediction in batches
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BSIZE):
    predicted_lables[i*BSIZE : (i+1)*BSIZE] = predict.eval(feed_dict={x: test_images[i*BSIZE : (i+1)*BSIZE],drop: 1.0})
                                                                                
# save results
np.savetxt('submission.csv', np.c_[range(1,len(test_images)+1), predicted_lables], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

#close session
sess.close()


