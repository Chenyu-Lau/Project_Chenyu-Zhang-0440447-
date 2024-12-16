Project Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset is divided into training and test sets, and the model is evaluated based on its performance on the test data.
The purpose of the project is to demonstrate how deep learning models, particularly CNNs, can be applied for image classification tasks.

Dataset
The dataset used for this project includes:

Training Set: 8000 images (4000 cats and 4000 dogs).
Test Set: 2000 images (1000 cats and 1000 dogs).
Single Prediction: A folder with test images for manual prediction.
Purpose of the Project
To implement a Convolutional Neural Network (CNN) using TensorFlow and Keras.
To preprocess image data for training and testing.
To train and evaluate the model's performance.
To predict the class (Cat or Dog) of unseen images.
Machine Learning Model
The following steps were implemented to build the CNN:

1. Data Preprocessing
Rescaling images using ImageDataGenerator.
Data augmentation techniques such as horizontal flip, shear, and zoom.
2. CNN Architecture
Input Layer: (64, 64, 3)
Convolutional Layers: 2 Conv2D layers with ReLU activation.
Pooling Layers: 2 MaxPooling2D layers.
Flattening Layer: To convert multi-dimensional data to one dimension.
Fully Connected Layer: Dense layer with 128 units.
Output Layer: Dense layer with 1 unit (Sigmoid activation for binary classification).
3. Training and Evaluation
Optimizer: Adam
Loss Function: Binary Cross-Entropy
Metrics: Accuracy
4. Prediction
The model predicts an unseen image as "Cat" or "Dog".

Results
Training Accuracy: ~84%
Validation Accuracy: ~77%
Test Accuracy: ~77%
The model successfully classifies images into two categories with reasonable accuracy.

Links
GitHub Repository: [Your GitHub Link Here]
Medium Post: [Your Medium Link Here]
Conclusion
This project demonstrates the application of Convolutional Neural Networks for binary image classification. The results are promising and show how CNNs can effectively solve classification problems in image data.

