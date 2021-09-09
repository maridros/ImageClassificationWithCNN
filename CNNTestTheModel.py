# Drosou Maria
# Department of Informatics And Computer Engineering, University of West Attica
# e-mail: cs151046@uniwa.gr
# A.M.: 151046

import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle  # we need it to read the data_batch file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import random

# loading the trained model & use it over test data
loaded_model = keras.models.load_model('OutputData/CIFAR_10_CNN.h5')

# loading test data
with open('InputData/test_batch', mode='rb') as file:
    batch = pickle.load(file, encoding='latin1')

batch_size = len(batch['data'])  # 10000 samples
channels = 3  # because it is in RGB

# input image dimensions
img_rows, img_cols = 32, 32

# reshape the data, so that they have the form of (batch_size
features = batch['data'].\
    reshape((batch_size, channels, img_cols, img_rows)).\
    transpose(0, 2, 3, 1)
labels = batch['labels']

# Print some info about the data
print('Loading data for test set completed.')
print('Features shape is:', features.shape)
print('So the number of samples is:', features.shape[0])

# loading labels names
with open('InputData/batches.meta', mode='rb') as file:
    meta = pickle.load(file, encoding='latin1')

labels_names = meta['label_names']
num_classes = len(labels_names)  # we have 10 classes

print("Representation of each class in test set:")
for i in range(0, 10):
    print('Class', i, '(' + labels_names[i] + '):', list(labels).count(i))

print('Pre-processing data...')

# convert RGB values from 0-255 to 0-1
x_test = features/255.0

print('Testing the loaded model...')

# test the model
y_test_predictions_vectorized = loaded_model.predict(x_test)
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)

# calculate and print some scores
acc_test = accuracy_score(labels, y_test_predictions)
pre_test = precision_score(labels, y_test_predictions, average='macro')
rec_test = recall_score(labels, y_test_predictions, average='macro')
f1_test = f1_score(labels, y_test_predictions, average='macro')

print('Accuracy score: {:.2f}'.format(acc_test))
print('Precision score: {:.2f}'.format(pre_test))
print('Recall score: {:.2f}'.format(rec_test))
print('F1 score: {:.2f}'.format(f1_test))


# plot confusion matrix
cm = confusion_matrix(labels, y_test_predictions)
df_cm = pd.DataFrame(cm, index=[i for i in labels_names], columns=[i for i in labels_names])
plt.figure(figsize=(10, 10))
sns.heatmap(df_cm, annot=True, fmt="d", square=True)
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Classifier label', fontsize=25)
plt.show()


# plot 4 images that correspond to each class, according to the CNN predictions
class_to_demonstrate = 0
random.seed(7)
while sum(y_test_predictions == class_to_demonstrate) > 4:

    # find the indices of the images which belong to this class
    tmp_idxs_to_use = np.where(y_test_predictions == class_to_demonstrate)

    # randomly choose 4 of the images of the class
    tmp_sampling = random.sample(list(tmp_idxs_to_use[0]), k=4)
    true_labels = [labels[i] for i in tmp_sampling]

    # create new plot window
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs[0, 0].imshow(features[tmp_sampling[0]])
    axs[0, 0].set_title(labels_names[true_labels[0]])
    axs[0, 1].imshow(features[tmp_sampling[1]])
    axs[0, 1].set_title(labels_names[true_labels[1]])
    axs[1, 0].imshow(features[tmp_sampling[2]])
    axs[1, 0].set_title(labels_names[true_labels[2]])
    axs[1, 1].imshow(features[tmp_sampling[3]])
    axs[1, 1].set_title(labels_names[true_labels[3]])

    tmp_title = labels_names[class_to_demonstrate]
    fig.suptitle('Classified as ' + tmp_title + ' (class ' + str(class_to_demonstrate) + ')')

    # show the plot
    plt.show()

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1


