# CS_project #
## Automated Defect Detection using Image Recognition in Manufacturing
This project aims to use image recognition techniques to detect defects in manufacturing products. The model learns to classify images into different categories, such as "defect" or "non-defect." The goal is to automate defect detection in manufacturing, with the positive outcome being increased productivity and cost-effectiveness, and the negative outcome being potential errors in classification.

**Code Overview**

The following sections provide an overview of the code and its functionality:

**Import Libraries**

This section imports the necessary libraries for data analysis, visualization, pre-processing, and building the CNN model.

**Importing Data**

This section imports the training and testing data from local directories and displays examples of defect and non-defect items.

**Show the Number of Data in Each Class**

This section calculates and displays the number of data samples in each class (OK and Defect) for both training and testing datasets.

**Preprocess the Images**

This section pre-processes the images by rescaling them and splitting the training dataset into a validation subset (20% of the training data). It uses the ImageDataGenerator class from Keras for data augmentation and scaling.

**Implementation of the CNN Model**

This section defines and summarizes the Convolutional Neural Network (CNN) model for automated defect detection. It consists of several layers, including convolutional, pooling, flatten, dense, and dropout layers.

**Choose the Optimizer and Loss Function**

This section configures the model for training by selecting the optimizer (Adam) and the loss function (binary crossentropy).

**Training the Model**

This section trains the CNN model using the training and validation datasets. It specifies the number of epochs and includes checkpoints and early stopping as callbacks to save the best model weights and prevent overfitting.

**Plot Loss and Accuracy Curves**

This section visualizes the accuracy and loss curves during the training process using the training and validation data.

**Test the Model**

This section evaluates the trained model on the test dataset, calculating the test loss and accuracy. It also generates predictions from the test dataset and displays a confusion matrix and a classification report to evaluate the model's performance.

**Test Case**

This section provides a list of test case images to be used for prediction.

**Conclusion**

This code demonstrates the implementation of an automated defect detection system using image recognition techniques in manufacturing. It trains a CNN model on a dataset of defect and non-defect images and achieves high accuracy in classifying new images. The model can be further improved by collecting more diverse data and fine-tuning the hyperparameters.
