# PlantVillage Dataset – Project Summary

## Data Loading
We started by loading the PlantVillage dataset directly from Google Drive. This ensured that all images and labels were properly accessible for further processing.
## Data Preprocessing
The following data preprocessing steps were conducted before training the network:
- Resize all images to the same resolution.
- Normalize pixel values.

## Dataset Analysis
Prior to the training procedure, we explored the given dataset:
- Inspected the dataset folder structure and class labels.
- Analyzed the distribution of classes across the dataset.

## Train-Validation Split
As part of the training process, we split the dataset into train and validation sets:
- 80% of the images were used for model training.
- The remaining 20% were allocated for evaluating the performance of our model.

## Model Development
We developed a simple baseline model, which was a CNN.
The network had an architecture that allowed extracting features from images of plant leaves and classifying them into disease classes.

## Training Pipeline
A comprehensive training pipeline was created for our model, which involved the following operations:
- Forward propagation.
- Backward propagation.
- Computation of loss.
- Updating weights by an optimizer.
