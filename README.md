# ImageClassificationWithCNN
CNN model training and testing on CIFAR-10 dataset.
## Requirements
- Python
- Keras
- MatPlotLib
- Numpy
- Pickle
- Scikit-learn
- Pandas
- Seaborn
## Code process and results
### Dataset
For this project data from CIFAR-10 dataset were used. These data were obtained from https://www.cs.toronto.edu/~kriz/cifar.html. This dataset containes 5 files-batches for the training phase, but in this project only data_batch_3 was used. For the testing phase test_batch was used. There is also batches.meta file which was used for loading labels' names.
### CNN model
The architecture of the CNN model implemented in the code is the following:
![CNN_diagram drawio](https://user-images.githubusercontent.com/89779679/132717432-a67e8649-a506-4a92-bcf1-ea2be7c65bc6.png)

### Code running
To run the code you need to have installed all the libraries which are referred in the Requirements above. Since you have done this, you need also to download the dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place data_batch_3, test_batch and batches.meta in InputData folder. After that for the training phase you run CNNTrainTheModel.py and for the testing phase you run CNNTestTheModel.py. In OutputData folder there is already a trained model, so that, if you like, you can run the testing phase without training first.
### Results
The CNN model classified the pictures of the test set wiht approximately 63% accuracy. It is not high enough, but it can be improved if we use all the bathces of the dataset in the training phase. Here you can see the confusion matrix which has been produced by the testing code:
![test_cm](https://user-images.githubusercontent.com/89779679/132719434-31ee726f-bd7b-4103-a879-6ab433d6a7b6.png)

