# Convolutional-NN-application
CNN is a powerful algorithm for image processing. These algorithms are currently the best algorithms we have for the automated processing of images. Many companies use these algorithms to do things like identifying the objects in an image. This is the reason why this is the tool selected in this work for handwritten digit recognition. We develop a full methodology for properly creating and training a CNN capable of recognizing any new input handwritten digit successfully. For this, we use a known database for the process of training and testing the Network.
To predict Digits from 0 through 9, the selected data set is the MNIST dataset. MNIST is an acronym that stands for the Modified National Institute of Standards and Technology that is commonly used for training various image processing systems. This dataset consists of 60,000 training images and 10,000 testing images, square 28×28 pixels grayscale images of handwritten single digits between 0 and 9.

MNIST is a dataset provided by Keras deep learning API. When the Keras API is called, there are four values returned namely  x_train, y_train, x_test, and y_test. They are representations for training and test datasets. 

Data Preprocessing: Some pre-processing operations of the data need to be performed. CNN will learn best from a dataset that does not contain any null values, has all numeric data, and is scaled. The dimension of the training data is (60000*28*28). One more dimension is needed for the CNN model so we reshape the matrix to shape (60000*28*28*1). Each image is 28X28 size, so there are 784 pixels. So, the output layer has 10 outputs, the hidden layer has 784 neurons and the input layer has 784 inputs. The dataset is then converted into float datatype.

Normalize the data: The pixel values for each image in the dataset are unsigned integers in the range 0-255 (0 means black and 255 means white). The output is a 2-dimensional matrix. We normalize the pixel values of grayscale images, e.g. rescale them from (0-255) to (0-1). This involves first converting the data type from unsigned integers to floats, and then dividing the pixel values by the maximum value. The target variable is one-hot encoded for further analysis. The target variable has a total of 10 classes (0-9)

Create the NN model: A CNN model generally consists of convolutional and pooling layers. The dropout layer is used to deactivate some of the neurons and while training, it reduces the over-fitting of the model. Activation functions are responsible for making decisions about whether to move forward. Based on activation functions, neurons fire up and the network moves forward. The model has two main aspects: the feature extraction comprised of convolutional and pooling layers, and the classifier that will make a prediction.
For the 1st part, we can start with a single convolutional layer (input layer) with a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer that takes the maximum value called MaxPooling2D. In this model, it is configured as a 2×2 pool size. Here, regularization happens. It is set to randomly exclude 20% of the neurons in the layer to avoid overfitting. For the hidden layers, we have 2 more convolutional layer with a filter size (3,3) and a greater number of filters (128) on each one.
The filter maps can then be flattened to provide features to the classifier. The flattened layer converts the 2D matrix data into a vector called Flatten. It allows the output to be fully processed by a standard fully connected layer.
Between the feature extractor and the output layer, we can add a fully connected (dense) layer to interpret the features, in this case with 100 nodes. Given that the problem is a multi-class classification task, we know that we will require an output layer with 10 nodes in order to predict the probability distribution of an image belonging to each of the 10 classes. This will also require the use of a softmax activation function. 

Except for the output layer, all layers will use the ReLU activation function and the He weight initialization scheme, both have proven to be best practices.

The optimization method will be stochastic gradient descent (SDG) optimizer with a conservative configuration for the with a learning rate of 0.01 and a momentum of 0.9. 
The categorical cross-entropy loss function will be optimized, suitable for multi-class classification, and we will monitor the classification accuracy metric, which is appropriate given we have the same number of examples in each of the 10 classes.
Visible Layer 1x28x28
Convolutional Layer 32 maps, 3×3
Max Pooling Layer 2×2
Convolutional Layer 64 maps, 3×3
Convolutional Layer 64 maps, 3×3
Max Pooling Layer 2×2
Flatten Layer 20%
Hidden Layer (100 neurons)
Output Layer (10 outputs)

Train the model: To start the training of the model, the model.fit() function of Keras is called. It takes the training data, validation data, epochs, and batch size as the parameter. During the process of fitting, the model will go through the dataset and understand the relations. It will learn throughout the process as many times as has been defined. 10 epochs are defined. 

Save the model: Once fit, we can save the final model to an H5 file by calling the save() function on the model and pass in the chosen filename.

Evaluate the Model: There exist several methods for evaluating the model, such that: 
•	Resubstituion
•	Hold-out
•	K-fold Cross-validation
•	LOOCV
•	Random Subsampling
•	Bootstrapping
As a default selection, the model will be evaluated using five-fold cross-validation. The value of k=5 was chosen to provide a baseline for both repeated evaluation and to not be so large as to require a long running time. 
Each test set will be 20% of the training dataset (about 12,000 examples). The training dataset is shuffled prior to being split, and the sample shuffling is performed each time. The test set for each fold will be used to evaluate the model both during each epoch of the training run.

Results: There are two key aspects to present: the diagnostics of the learning behavior of the model during training and the estimation of the model performance. These can be implemented using separate functions.
First, the diagnostics involve creating a line plot showing model performance on the train and test set during each fold of the k-fold cross-validation. These plots are valuable for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset.
Two subplots are created, one for loss and one for accuracy. Blue lines will indicate model performance on the training dataset and orange lines will indicate performance on the hold-out test dataset. 
Next, the classification accuracy scores collected during each fold can be summarized by calculating the mean and standard deviation. This provides an estimate of the average expected performance of the model trained on this dataset, with an estimate of the average variance in the mean.

Making a Prediction: The model assumes that new images are grayscale, that they have been aligned so that one image contains one centered handwritten digit, and that the size of the image is square with the size 28×28 pixels. The loaded image can then be resized to have a single channel and represent a single sample in a dataset. The pixel values are prepared in the same way as the pixel values were prepared for the training dataset when fitting the final model, in this case, normalized.
