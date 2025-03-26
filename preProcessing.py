import numpy as np
import os
import cv2 as cv
from sklearn.model_selection import train_test_split


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


def main():

    image_path = 'source_images'
    defect_path = os.path.join(image_path, 'def_front')
    ok_path = os.path.join(image_path, 'ok_front')

    # Loading images and labelling them
    defect_images, defect_labels = loadImages(defect_path, 1)
    ok_images, ok_labels = loadImages(ok_path, 0)

    images = np.array(defect_images + ok_images, dtype=np.float32) / 255.0
    labels = np.array(defect_labels + ok_labels)

    print('Loading and labelling of images are completed.')


if __name__ == '__main__':
    main()


