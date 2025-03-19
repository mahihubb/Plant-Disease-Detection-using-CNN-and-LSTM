# Plant Disease Detection using CNN and LSTM

## Overview
This project aims to detect plant diseases using deep learning techniques, specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The goal is to classify plant leaves as healthy or diseased, enabling early detection and prevention of crop loss.

## Dataset
The dataset used for training and evaluation consists of labeled images of plant leaves, categorized into different disease types and a healthy class. The dataset was sourced from:
- **PlantVillage Dataset**: A widely used dataset containing images of healthy and diseased plant leaves.
- **Custom Augmented Dataset**: Additional images generated through data augmentation techniques such as rotation, flipping, and contrast adjustment.

## Models Used
### CNN Model
The **CNN model** extracts spatial features from plant leaf images and learns patterns that distinguish between healthy and diseased leaves. It includes multiple convolutional layers, max-pooling layers, and fully connected layers for classification.

**Model Architecture:**
- Input Layer: Preprocessed plant leaf images
- Convolutional Layers: Feature extraction
- Max-Pooling Layers: Dimensionality reduction
- Fully Connected Layers: Classification
- Output Layer: Softmax activation for multi-class classification

📌 **Model Download Link**: [CNN_model.keras](https://drive.google.com/file/d/1kXH7bfH57Bp6nYDHB1sbPCYcpu-UYgA6/view?usp=sharing)

### LSTM Model
The **LSTM model** is used to analyze sequential dependencies in the extracted features from CNN. It enhances classification accuracy by capturing temporal patterns in feature maps.

**Model Workflow:**
- CNN extracts feature representations from images
- LSTM processes the extracted feature sequences
- Fully connected layers perform classification
- Softmax activation outputs the final disease category

## Methodology
1. **Data Preprocessing:** Image resizing, normalization, and augmentation
2. **Model Training:**
   - Train CNN separately on image data
   - Train LSTM on feature representations extracted from CNN
3. **Evaluation:** Model accuracy, precision, recall, and F1-score metrics
4. **Deployment:** Model saved in `.keras` format for inference

## Outcomes
- Achieved high accuracy in classifying plant diseases
- Improved disease detection efficiency compared to traditional manual methods
- Provided a scalable and automated solution for farmers and researchers

## Future Work
- Enhance model performance with additional training data
- Implement real-time disease detection using mobile applications
- Integrate explainable AI techniques for better interpretability

---
✉️ **For queries and contributions, feel free to reach out!**

