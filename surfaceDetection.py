import os
from preProcessing import loadData, createDataset
from sklearn.model_selection import train_test_split
from Models import buildAlexNet, buildLeNet, buildResNet
from evaluations import saveMetrics, evaluateModel
from keras.callbacks import EarlyStopping
from explainability import make_gradcam_heatmap, display_gradcam
import tensorflow as tf
import numpy as np

def explain_model(model, sample_image):

    img_array = np.expand_dims(sample_image, axis=0)  # Now shape is (1, 128, 128, 3)

    print('Loading completed\n'
          'Starting GradCam ')

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv2d')
    display_gradcam(sample_image, heatmap)


def main():

    print('Loading of data should take place here: ')

    # Paths to source folders
    source_folder = 'source_images'
    defect_path = os.path.join(source_folder, 'def_front')
    ok_path = os.path.join(source_folder, 'ok_front')
    model = tf.keras.models.load_model('Models/trained_LeNet_noAug.h5')


    # Loading images and labelling them
    image_size = (128, 128)
    images, labels = loadData(defect_path, ok_path, size=image_size)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    train_ds = createDataset(X_train, y_train)

    # Building the different models
    input_shape = (image_size[0], image_size[1], 3)
    sample_image = X_test[0]

    #model = buildLeNet(input_shape=input_shape)
    #model = buildAlexNet(input_shape=input_shape)
    model = buildResNet(input_shape=input_shape)


    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #history = model.fit(train_ds, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

    model.save('Models/trained_ResNet_noAug.h5')

    evaluateModel(model, X_test, y_test)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    saveMetrics('Resnet: No Augmentations', history, test_acc, test_loss)


if __name__ == '__main__':
    main()
