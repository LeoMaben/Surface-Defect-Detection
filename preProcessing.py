import numpy as np
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import matplotlib.pyplot as plt


def loadImages(folder_path:os.path, label:int):
    """

    Loads the images and labels from the source paths

    :param folder_path: Path for the images
    :param label: Label indicating if defective (1) or not (0)
    :return:
    """


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

    augmented_images = [image]
    for _ in range(10):
        image_aug1 = tf.image.random_flip_left_right(image)
        image_aug2 = tf.image.random_flip_up_down(image)
        image_aug3 = tf.image.random_brightness(image, max_delta=0.2)
        image_aug4 = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        augmented_images.extend([image_aug1, image_aug2, image_aug3, image_aug4])
    labels = [label] * len(augmented_images)

    return tf.data.Dataset.from_tensor_slices((augmented_images, labels))



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

    # Paths for images
    image_path = 'source_images'
    defect_path = os.path.join(image_path, 'def_front')
    ok_path = os.path.join(image_path, 'ok_front')

    # Loading images and labelling them
    defect_images, defect_labels = loadImages(defect_path, 1)
    ok_images, ok_labels = loadImages(ok_path, 0)

    # Changing to the required np format
    images = np.array(defect_images + ok_images, dtype=np.float32) / 255.0
    labels = np.array(defect_labels + ok_labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

    # print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


    train_ds = train_ds.flat_map(imageAugementation)
    train_ds = train_ds.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32)

    checkAugmentedImages(train_ds)

    num_images = sum(len(images) for images, labels in train_ds)
    print(f"Total number of images: {num_images}")



if __name__ == '__main__':
    main()


