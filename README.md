# Surface-Defect-Detection

The goal of this project is to test out the various methods for defect detection.
Main goal is to further my understanding of how the inner working of these models function

## Roadmap/Plan: 
- ~~Load and resize the data~~
- ~~Perform data augmentation to have more training~~ and testing samples 
- ~~Split the data into train, test and hold-out~~ set 
- Create 3 different CNN models:
  - ~~A simple model with only two convolutions emulating the architecture of LeNet~~ 
  - A ResNet Model 
  - GradCam to Visualize the explainability
  - Anomaly Detection
  - Evaluation
- Compare the results 
- Final Docs

## Steps Taken:
1) Data Pipeline for loading the images and their respective labels while splitting into train-test sets
2) Data Augmentation Pipeline for larger data samples for training the model
3) Using a LeNet Architecture to get initially ~85% Accuracy without data augmentation and increasing to ~98% Acucuracy
   on the test set with the use of Early Stopping and L2 Batch Normalization to avoid over-fitting
4) 


