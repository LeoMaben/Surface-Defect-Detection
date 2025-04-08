import os
from preProcessing import loadData, createDataset
from sklearn.model_selection import train_test_split
from Models import buildAlexNet, buildLeNet, buildResNet, buildCustomResNet
from evaluations import plotTrainingHistory
from keras.callbacks import EarlyStopping

def main():

    print('Loading of data should take place here: ')

    # Paths to source folders
    source_folder = 'source_images'
    defect_path = os.path.join(source_folder, 'def_front')
    ok_path = os.path.join(source_folder, 'ok_front')

    # Loading images and labelling them
    images, labels = loadData(defect_path, ok_path, size=(256, 256))

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    train_ds = createDataset(X_train, y_train)

    # Building the different models
    #model = buildLeNet()
    model = buildAlexNet()

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #history = model.fit(train_ds, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"The Accuracy of the LeNet model is : {test_acc}")

    plotTrainingHistory(history)


if __name__ == '__main__':
    main()