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
            image = cv.resize(image, (128, 128))
            images.append(image)
            labels.append(label)

    return images, labels


def loadData(defect_path, ok_path):
    # Loading images and labelling them
    defect_images, defect_labels = loadImages(defect_path, 1)
    ok_images, ok_labels = loadImages(ok_path, 0)

    # Changing to the required np format
    images = np.array(defect_images + ok_images, dtype=np.float32) / 255.0
    labels = np.array(defect_labels + ok_labels)

    return images, labels


def augmentImage(image):
    """Apply random augmentations to a single image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

def augmentImages(image, label, num_augmentations=10):
    """Generate augmented images."""
    augmented_images = [image]
    for _ in range(num_augmentations - 1):  # -1 because the original image is already added
        augmented_image = augmentImage(image)
        augmented_images.append(augmented_image)

    # Create a dataset of augmented images and corresponding labels
    augmented_labels = [label] * len(augmented_images)
    return tf.data.Dataset.from_tensor_slices((augmented_images, augmented_labels))



def count_labels(tensor_dataset):

    count_defective = 0
    count_ok = 0

    for image, label in tensor_dataset:
        if label.numpy() == 1:
            count_defective += 1
        else:
            count_ok += 1

    print(f"Total number of labels: {count_ok + count_defective}\n"
          f"The ok labels are {count_ok} and defective labels are {count_defective}")



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


def createDataset(images, labels, batch_size=32, num_augmentations=10):
    """Create a TensorFlow dataset from images and labels with augmentation."""
    # Convert the images and labels to a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Apply augmentation to each image
    dataset = dataset.flat_map(lambda image, label: augmentImages(image, label, num_augmentations))

    # Shuffle, batch, and prefetch the dataset for efficient training
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def main():

    # Paths for images
    image_path = 'source_images'
    defect_path = os.path.join(image_path, 'def_front')
    ok_path = os.path.join(image_path, 'ok_front')

    # Loading Data into respective numpy arrays
    images, labels = loadData(defect_path, ok_path)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

    # print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    train_ds = createDataset(X_train, y_train)



if __name__ == '__main__':
    main()


