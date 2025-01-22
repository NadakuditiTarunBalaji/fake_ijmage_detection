# Description: This script is used to train the model using MobileNetV2 architecture. The model is trained on the dataset
# containing real and fake face images. The model is then saved as a .keras file. The training history is visualized
# using matplotlib. The model is then used in the app.py script to make predictions on new images.
#----------------------------------------------
import numpy as np
import pandas as pd
from keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, InceptionV3
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import cv2

# Define paths
real = "D:\\delete\\DeepFake-Detection\\archive\\real_and_fake_face\\training_real"
fake = "D:\\delete\\DeepFake-Detection\\archive\\real_and_fake_face\\training_fake"
dataset_path = "D:\\delete\\DeepFake-Detection\\archive\\real_and_fake_face"

# Data Augmentation
# Load the dataset and apply data augmentation techniques to the training data using ImageDataGenerator class from Keras API.
data_with_aug = ImageDataGenerator(
    # Data Augmentation Techniques (Random Flips) and Normalization (Rescaling) applied to the training data 
    horizontal_flip=True,
    vertical_flip=False,
    # Rescale the pixel values of the images to the range [0, 1]
    rescale=1./255,
    # Split the dataset into training and validation sets with 80% of the data used for training and 20% for validation.
    validation_split=0.2
)
# Load the dataset and apply data augmentation techniques to the training data using ImageDataGenerator class from Keras API.
train = data_with_aug.flow_from_directory(
    # Load the dataset from the specified path and set the class mode to binary.
    dataset_path,
    # Set the target size of the images to (224, 224) and the batch size to 32.
    class_mode="binary",
    # Set the target size of the images to (224, 224) and the batch size to 32.
    target_size=(224, 224),
    # Set the batch size to 32 and the subset to training.
    batch_size=32,
    # Set the subset to training.
    subset="training"
)
# Load the dataset and apply data augmentation techniques to the validation data using ImageDataGenerator class from Keras API.
val = data_with_aug.flow_from_directory(
    # Load the dataset from the specified path and set the class mode to binary.
    dataset_path,
    # Set the target size of the images to (224, 224) and the batch size to 32.
    class_mode="binary",
    # Set the target size of the images to (224, 224) and the batch size to 32.
    target_size=(224, 224),
    # Set the batch size to 32 and the subset to validation.
    batch_size=32,
    # Set the subset to validation.
    subset="validation"
)

# Model Building (MobileNetV2 example)
# Define the input layer with the shape of the image (224, 224, 3) and initialize MobileNetV2 without top layers. 
input_layer = Input(shape=(224, 224, 3))

# Initialize MobileNetV2 without top layers
# MobileNetV2 is a convolutional neural network that is trained on more than a million images from the ImageNet database. 
mnet = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# Define the model architecture by adding the input layer and the MobileNetV2 model to the output layer. 
# The output layer consists of GlobalAveragePooling2D, Dense, BatchNormalization, and Dropout layers.
x = mnet(input_layer)
#Global Average Pooling Layer 
x = GlobalAveragePooling2D()(x)
#Dimensionality Reduction Prevent Overfitting
x = Dense(512, activation="relu")(x)
#Batch Normalization Layer
x = BatchNormalization()(x)
#Dropout Layer 
x = Dropout(0.3)(x)
#Dense Layer 
x = Dense(128, activation="relu")(x)
#Batch Normalization Layer 
x = Dropout(0.1)(x)
# The output layer consists of a Dense layer with 2 units and softmax activation function.
output = Dense(2, activation="softmax")(x)
# Define the model by specifying the input layer and the output layer.
model = Model(inputs=input_layer, outputs=output)

# Freeze MobileNetV2 layers
model.layers[1].trainable = False

# Compile the model
# Compile the model using the sparse categorical crossentropy loss function, Adam optimizer, and accuracy metric. 
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Learning Rate Scheduler
def scheduler(epoch):
    # Set the learning rate to 0.001 for the first 2 epochs, 0.0001 for the next 13 epochs, and 0.00001 for the remaining epochs.
    if epoch <= 2:
        # Set the learning rate to 0.001 for the first 2 epochs.
        return 0.001
    elif epoch > 2 and epoch <= 15:
        # Set the learning rate to 0.0001 for the next 13 epochs.
        return 0.0001
    # Set the learning rate to 0.00001 for the remaining epochs.
    else:
        # Set the learning rate to 0.00001 for the remaining epochs.
        return 0.00001
# Define the learning rate scheduler using the LearningRateScheduler callback.
lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)
#----------------------------------------------
# Train the model
hist = model.fit(
    train,
    # Set the number of epochs to 20 and the learning rate scheduler as a callback.
    epochs=20,
    # Set the learning rate scheduler as a callback.
    callbacks=[lr_callbacks],
    # Set the validation data to the validation set.
    validation_data=val
)

# Save the model as .keras
# Save the model as a .keras file using the save method.
model.save('face_detection_model.keras')
#----------------------------------------------

# Visualizing the accuracy and loss
epochs = 20
# Get the training and validation loss and accuracy from the training history.
train_loss = hist.history['loss']
#   Get the training and validation loss and accuracy from the training history.
val_loss = hist.history['val_loss']
#  Get the training and validation loss and accuracy from the training history.
train_acc = hist.history['accuracy']
#  Get the training and validation loss and accuracy from the training history.
val_acc = hist.history['val_accuracy']
# Create a range of epochs for the x-axis.
xc = range(epochs)
plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Val'])
# Create a range of epochs for the x-axis.
plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Val'])
# Show the plot
plt.show()
