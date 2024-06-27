
# Gem Classifier Project by Sapir Dahan

## Overview

This project aims to classify 75 different types of gem images using a neural network model. Given the limited dataset, transfer learning is employed using a pre-trained ResNet50 model. The project is implemented using TensorFlow.

## Project Structure

The project consists of two main parts:
1. **Training**: Training the gem classifier model using transfer learning.
2. **Inference**: Using the trained model to classify new gem images.

## Training

The training process involves the following steps:
1. **Examine and Understand the Data**: Analyze the dataset to understand its structure and contents.
2. **Build an Input Pipeline**: Prepare the data for training, including data augmentation and batching.
3. **Compose the Model**:
   - Load the pre-trained ResNet50 base model.
   - Stack the classification layers on top of the base model.
4. **Train the Model**:
   - Feature Extraction: Freeze the pre-trained layers and train the new classification layers.
   - Fine-Tuning: Unfreeze some of the pre-trained layers and jointly train them with the new layers.
5. **Evaluate the Model**: Assess the performance of the model on the test set.

### Hyperparameters
- `BATCH_SIZE`: 32
- `IMG_SIZE`: (224, 224)
- `test_split`: 0.1
- `validation_split`: 0.1
- `initial_epochs`: 300
- `fine_tune_epochs`: 2000
- `base_learning_rate`: 0.0001

## Inference

The inference process involves the following steps:
1. **Load the Model and Class Names**:
   - Load the pre-trained model using `tf.keras.models.load_model`.
   - Load the class names from a pickle file.
2. **Process the Image**:
   - Load and resize the input image to (224, 224).
   - Convert the image to a NumPy array and reshape it for model input.
3. **Make Predictions**:
   - Use the model to predict the class of the input image.
   - Extract the top 5 predictions and their confidence levels.
4. **Visualize the Results**:
   - Create a bar plot showing the top 5 predicted classes and their confidence levels.
   - Display the input image.

## How to Use

### Training
1. Ensure you have TensorFlow installed.
2. Prepare your dataset and adjust the hyperparameters if necessary.
3. Run the `Gem_Classifier_Train.ipynb` notebook to train the model.

### Inference
1. Ensure you have TensorFlow and the necessary dependencies installed.
2. Place your input images in the specified directory.
3. Run the `Gem_Classifier_Inference.ipynb` notebook to classify new gem images.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib
- Pillow
- Pickle
