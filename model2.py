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
data_with_aug = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1./255,
    validation_split=0.2
)

train = data_with_aug.flow_from_directory(
    dataset_path,
    class_mode="binary",
    target_size=(224, 224),
    batch_size=32,
    subset="training"
)

val = data_with_aug.flow_from_directory(
    dataset_path,
    class_mode="binary",
    target_size=(224, 224),
    batch_size=32,
    subset="validation"
)

# Model Building (MobileNetV2 example)
input_layer = Input(shape=(224, 224, 3))

# Initialize MobileNetV2 without top layers
mnet = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

x = mnet(input_layer)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)

output = Dense(2, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)

# Freeze MobileNetV2 layers
model.layers[1].trainable = False

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Learning Rate Scheduler
def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001

lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train the model
hist = model.fit(
    train,
    epochs=20,
    callbacks=[lr_callbacks],
    validation_data=val
)

# Save the model as .keras
model.save('face_detection_model.keras')

# Visualizing the accuracy and loss
epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Val'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Val'])

plt.show()
