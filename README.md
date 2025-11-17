 ♻️ Smart Waste Classification System using CNN and Deep Learning for a Sustainable Environment

 🧩 Problem Statement

Waste management has become a major challenge in achieving environmental sustainability.  
Improper waste segregation leads to pollution, recycling inefficiency, and health hazards.  
Manual segregation is time-consuming and error-prone.  
This project aims to develop a **Deep Learning-based Smart Waste Classification System** using **Convolutional Neural Networks (CNN)** that can automatically classify waste as **Organic** or **Recyclable** from images.  
By automating waste sorting, the project contributes to building **smart and sustainable cities**.

🎯 Objectives

- Build a **CNN model** capable of classifying waste images accurately.  
- Use **Deep Learning** to analyze image patterns and automate waste recognition.  
- Clean and preprocess the dataset (remove errors, duplicates, rename files).  
- Evaluate model performance and visualize results.  
- Promote **sustainability and environmental responsibility** through AI-based automation.  

Week 1: Dataset Cleaning and Preparation
 ✅ Tasks Completed
- Verified image folders and class distribution  
- Removed **326 duplicate images** using hashing  
- Checked and removed corrupted image files  
- Ensured all images had proper `.jpg` extensions  
- Organized clean dataset into **Train** and **Test** directories  
 🧠 Output Summary
- **TRAIN/O** → 12,565 images  
- **TRAIN/R** → 9,999 images  
- **TEST/O** → 1,400 images  
- **TEST/R** → 1,112 images  
- **Duplicate Images Removed:** 326  
- **Corrupted Images Found:** 0  

 📊 Dataset Details

**Source:** [Kaggle – Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)

**Dataset Structure:** 
DATASET/
│
├── TRAIN/
│   ├── O/   → Organic Waste Images  
│   └── R/   → Recyclable Waste Images  
│
└── TEST/
    ├── O/   → Organic Waste Images  
    └── R/   → Recyclable Waste Images

Week 2 – Model Development & Training (CNN – Image Classification)
✅ Task Overview

In Week 2, the goal was to build and train a Convolutional Neural Network (CNN) to classify images into two categories related to sustainability (example: recyclable vs. non-recyclable).

🧠 Objectives of Week 2

Objective	Status:
Load dataset into training/testing sets	✅ Completed
Build a CNN model using TensorFlow & Keras	✅ Completed
Train the model and visualize accuracy/loss graphs	✅ Completed
Save trained model for future prediction (Week 3)	✅ Completed

📂 Folder Structure
Week2/
│
├── dataset/
│   ├── train/
│   │   ├── recyclable/
│   │   └── non_recyclable/
│   └── test/
│       ├── recyclable/
│       └── non_recyclable/
│
├── week2_cnn_training.ipynb  (Jupyter Notebook)
├── sustainable_image_cnn.keras  (saved model)
└── README.md

🔧 Technologies Used:
*Tool / Library	Purpose
*Python	Programming
*TensorFlow / Keras	CNN Model building & training
*Matplotlib	Accuracy & Loss Visualization
*Jupyter Notebook	Development Environment


🔁 Steps Performed
✅ Step 1: Load dataset using ImageDataGenerator

Images are automatically resized to 224 × 224 and normalized.

✅ Step 2: Build CNN model

Used Conv2D, MaxPooling2D, Flatten, Dense, Dropout.

✅ Step 3: Train the model

✅ Step 4: Plot Accuracy & Loss Graphs

Graphs show improvement across epochs.

📈 Output Graphs (Generated in Notebook)

Model Training Accuracy vs Validation Accuracy

Model Training Loss vs Validation Loss

(Graphs visible in notebook output.)

💾 Saved Model

The model is saved in the new recommended Keras format:

model.save("sustainable_image_cnn.keras")

📘 Week 3 – Model Evaluation & Deployment
Project: Smart Waste Classification System using CNN & Deep Learning
🔹 Week 3 Objectives

During Week 3, the goal was to evaluate the CNN model, analyze misclassifications, and deploy the final model using Streamlit.

✅ Tasks Completed in Week 3
1️⃣ Model Evaluation

After training the CNN model, detailed evaluation was performed using:

✔ Confusion Matrix

Organic correctly predicted: 1328

Organic misclassified as Recyclable: 72

Recyclable correctly predicted: 952

Recyclable misclassified as Organic: 160

✔ Classification Report
Metric	Organic (0)	Recyclable (1)
Precision	0.89	0.93
Recall	0.95	0.86
F1-Score	0.92	0.89

Overall Accuracy: 91%

Total Test Images: 2512

Misclassified Images Stored: 232

2️⃣ Misclassified Image Storage

A script was executed to identify wrongly classified images and store them for analysis.

📁 Folder Created: misclassified_images/
📌 Total Misclassified Images: 232

This helps understand model weaknesses and evaluate improvement areas.

3️⃣ Single Image Prediction

A test image was passed through the trained model to verify real-time prediction.

Example Output:

Prediction: Recyclable (R)

Probability: 0.638

Image displayed with predicted label

This validates the correctness of the trained CNN model.

4️⃣ Deployment Using Streamlit

The final model was deployed as a Streamlit web application.

📄 File: app.py
🎯 Features:

Upload an image (JPG/PNG)

Model predicts:

Organic (O) or Recyclable (R)

Shows prediction + confidence score

Displays uploaded image

📌 Command to Run App (in Terminal):

streamlit run app.py
