import numpy as np
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


def loadImages(folder_path:os.path, label:int):
    images, labels = [], []

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        image = cv.imread(img_path)
        if image is not None:
            image = cv.resize(image, (256, 256))
            images.append(image)
            labels.append(label)

    return images, labels


def imageAugementation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label


def checkAugmentedImages(tensor_dataset):
    images, labels = next(iter(tensor_dataset))
    images = np.array(images)
    labels = np.array(labels)


    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i])
        class_label = 'Non Defective' if labels[i] == 0 else 'Defective'
        plt.xlabel(class_label)
    plt.suptitle('Test Images')
    plt.show()


def main():

    image_path = 'source_images'
    defect_path = os.path.join(image_path, 'def_front')
    ok_path = os.path.join(image_path, 'ok_front')

    # Loading images and labelling them
    defect_images, defect_labels = loadImages(defect_path, 1)
    ok_images, ok_labels = loadImages(ok_path, 0)

    images = np.array(defect_images + ok_images, dtype=np.float32) / 255.0
    labels = np.array(defect_labels + ok_labels)

    X_train, X_test, y_train, y_test = train_test_split(images ,labels, test_size=0.2, stratify=labels, random_state=42)

    # print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.map(imageAugementation).batch(32).shuffle(100)
    test_ds = test_ds.batch(32)

    checkAugmentedImages(train_ds)

    num_batches = sum(1 for _ in train_ds)
    print(f"Total number of batches: {num_batches}")

    num_images = sum(len(images) for images, _ in train_ds)
    print(f"Total number of images: {num_images}")


if __name__ == '__main__':
    main()


