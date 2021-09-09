# Drosou Maria
# Department of Informatics And Computer Engineering, University of West Attica
# e-mail: cs151046@uniwa.gr
# A.M.: 151046

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import pickle  # we need it to read the data_batch file
from sklearn.model_selection import train_test_split
import random


# load train data
with open('InputData/data_batch_3', mode='rb') as file:
    batch = pickle.load(file, encoding='latin1')

batch_size = len(batch['data'])  # 10000 samples
channels = 3  # because it is in RGB

# input image dimensions
img_rows, img_cols = 32, 32

# reshape the data, so that they have the correct form
features = batch['data'].\
    reshape((batch_size, channels, img_cols, img_rows)).\
    transpose(0, 2, 3, 1)
labels = batch['labels']

# Print some info about the data
print('Loading data for training and validation set completed.')
print('Features shape is:', features.shape)
print('So the number of samples is:', features.shape[0])

# load labels names
with open('InputData/batches.meta', mode='rb') as file:
    meta = pickle.load(file, encoding='latin1')

labels_names = meta['label_names']
num_classes = len(labels_names)  # we have 10 classes

# Print some info about the classes
print('Loading metadata (labels names) completed.')
print("There are", num_classes, "classes.")
print("Representation of each class:")
for i in range(0, 10):
    print('Class', i, '(' + labels_names[i] + '):', list(labels).count(i))

# plot 4 images of each class
class_to_demonstrate = 0
random.seed(7)
while sum(np.array(labels) == class_to_demonstrate) > 4:

    # find the indices of the images which belong to this class
    tmp_idxs_to_use = np.where(np.array(labels) == class_to_demonstrate)

    # randomly choose 4 of the images of the class
    tmp_sampling = random.sample(list(tmp_idxs_to_use[0]), k=4)

    # create new plot window
    plt.figure()

    # plot the 4 images
    plt.subplot(221)
    plt.imshow(features[tmp_sampling[0]])
    plt.subplot(222)
    plt.imshow(features[tmp_sampling[1]])
    plt.subplot(223)
    plt.imshow(features[tmp_sampling[2]])
    plt.subplot(224)
    plt.imshow(features[tmp_sampling[3]])
    tmp_title = labels_names[class_to_demonstrate]
    plt.suptitle(tmp_title + ' (class ' + str(class_to_demonstrate) + ')')

    # show the plot
    plt.show()

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1

print('Pre-processing data and producing train and validation set...')

# convert RGB values from 0-255 to 0-1
features = features/255.0

# convert class vectors to binary class matrices
labels = keras.utils.to_categorical(labels, num_classes)

# split data to train and validation set
x_train, x_validate, y_train, y_validate = train_test_split(features, labels, test_size=0.03, random_state=0)

print('Number of train samples:', x_train.shape[0])
print("Representation of each class in train set:")
tmp_y_train = np.argmax(y_train, axis=1)
for i in range(0, 10):
    print('Class', i, '(' + labels_names[i] + '):', list(tmp_y_train).count(i))


print('Number of validation samples:', x_validate.shape[0])
print("Representation of each class in validation set:")
tmp_y_validate = np.argmax(y_validate, axis=1)
for i in range(0, 10):
    print('Class', i, '(' + labels_names[i] + '):', list(tmp_y_validate).count(i))


print('Producing the CNN model...')
# here we define and load the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# print model summary
model.summary()

epochs = 40

print('Start fitting model parameters. Epochs =', str(epochs) + '.')

# fit model parameters, given a set of training data
model.fit(x_train, y_train,
          batch_size=32,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validate, y_validate))

print('Fitting completed.')

# calculate some common performance scores
score = model.evaluate(x_validate, y_validate, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# saving the trained model
model_name = 'CIFAR_10_CNN.h5'
path = 'OutputData/'
model.save(path+model_name)

print('Model saved as', model_name + '.')
