import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

def buildLeNet(input_shape=(128, 128, 3)):
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


def buildUNet():
    return None

def buildResNet():
    return None

def main():
    print('Loading of data should take place here: ')


if __name__ == '__main__':
    main()

