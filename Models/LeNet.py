import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preProcessing import loadImages


def plotTrainingHistory(history):

    plt.figure(figsize=(12,5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label= 'Train Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Train Loss')
    plt.plot(history.history['val_loss'], label = 'Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def buildLeNet(input_shape=(128, 128, 3)):
    '''
    Simple straightforward LeNet architecture for comparison of binary classification
    :param input_shape:
    :return:
    '''
    model = tf.keras.Sequential([
        # Layer 1
        layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPool2D((2, 2)),
        # Layer 2
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Flattening the values
        layers.Flatten(),
        layers.Dense(120, activation='relu'), # Layer 3
        layers.Dense(84, activation='relu'), # Layer 4
        layers.Dense(1, activation='sigmoid')  # Layer 5 Sigmoid for classification?
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



def buildUNet():
    return None

def buildResNet():
    return None

def main():

    print('Loading of data should take place here: ')

    # Paths to source folders
    source_folder = '../source_images'
    defect_path = os.path.join(source_folder, 'def_front')
    ok_path = os.path.join(source_folder, 'ok_front')

    # Loading images and labelling them
    defect_images, defect_labels = loadImages(defect_path, 1)
    ok_images, ok_labels = loadImages(ok_path, 0)

    # Changing to the required np format
    images = np.array(defect_images + ok_images, dtype=np.float32) / 255.0
    labels = np.array(defect_labels + ok_labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

    model = buildLeNet()

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"The Accuracy of the LeNet model is : {test_acc}")

    plotTrainingHistory(history)



if __name__ == '__main__':
    main()

