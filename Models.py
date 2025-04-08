import tensorflow as tf
from keras import layers
from keras import regularizers
from keras.applications import ResNet50



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
        layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.01)), # Layer 3
        layers.Dense(84, activation='relu'), # Layer 4
        layers.Dense(1, activation='sigmoid')  # Layer 5 Sigmoid for classification?
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def buildAlexNet(input_shape=(256, 256, 3), num_of_classes = 1):

    model = tf.keras.models.Sequential([
        layers.Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_of_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



def buildResNet(input_shape = (256, 256, 3)):

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def buildCustomResNet(input_shape=(256, 256, 3), num_of_classes=1):
    inputs = tf.keras.Input(shape=input_shape)




