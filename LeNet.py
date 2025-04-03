import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from preProcessing import loadImages, loadData, augmentImages, createDataset
from keras.callbacks import EarlyStopping
from keras import regularizers
from evaluations import plotTrainingHistory



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

def main():

    print('Loading of data should take place here: ')

    # Paths to source folders
    source_folder = 'source_images'
    defect_path = os.path.join(source_folder, 'def_front')
    ok_path = os.path.join(source_folder, 'ok_front')

    # Loading images and labelling them
    images, labels = loadData(defect_path, ok_path)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    train_ds = createDataset(X_train, y_train)

    model = buildLeNet()

    #history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_ds, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"The Accuracy of the LeNet model is : {test_acc}")

    plotTrainingHistory(history)

if __name__ == '__main__':
    main()

