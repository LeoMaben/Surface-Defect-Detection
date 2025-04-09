import os.path

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import numpy as np
import json


def evaluateModel(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype('int32')

    print('Confusion Matrix: ')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print('Classification Report: Precision, Recall, F1 Score and Support')
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test,y_pred_probs)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred_probs)
    print(f'The AUC Curve is: {auc}')

    plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {auc:.2f}')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def plotTrainingHistory(history):

    plt.figure(figsize=(12,5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label= 'Train Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Train Loss')
    plt.plot(history.history['val_loss'], label = 'Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def saveMetrics(model_name, history, test_acc, test_loss,
                file='Results/metrics.json'):



    data = {
        model_name: {
            f'final accuracy': float(test_acc),
            f'final loss': float(test_loss),
            f'train accuracy': float(history.history['accuracy'][-1]),
            f'validation accuracy': float(history.history['val_accuracy'][-1])
        }
    }

    if os.path.exists(file):
        with open(file, 'r') as f:
            info = json.load(f)
    else:
        info = {}

    info.update(data)
    with open(file, 'w') as f:
        json.dump(info, f, indent=4)



