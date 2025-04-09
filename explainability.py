import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import cv2 as cv

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    # Basically maps the input image to the activiations of the last layer and the output
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Comparing the gradient of the top predicition with the last layer
    with tf.GradientTape() as tape:
        last_conv_layer_name, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_name)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1 ,2)) # Mean Intensity

    # Weighing the conv output by importance
    last_conv_layer_name = last_conv_layer_name[0]
    heatmap = last_conv_layer_name @ pooled_grads[..., tf.newaxis]
    heatmap - tf.squeeze(heatmap)

    # Normalization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(image, heatmap, alpha=0.4):
    img = np.uint8(255 * image)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colour = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    overlay = cv.addWeighted(img, 1 - alpha, heatmap_colour, alpha, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Grad-CAM Visualization for explaining the main features')
    plt.show()
