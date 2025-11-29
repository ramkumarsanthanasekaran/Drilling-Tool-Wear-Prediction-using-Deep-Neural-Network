# Drilling-Tool-Wear-Prediction-using-Deep-Neural-Network
Drilling Tool Wear Prediction using Deep Neural Network

This project develops a Deep Neural Network (DNN) model to predict drilling tool wear using machining parameters and sensor data.  
The dataset contains real experimental drilling measurements and associated wear states.

This repository provides a clean, industry-ready machine learning workflow suitable for academic submissions, applied research, and GitHub portfolio projects.

## 🔍 Project Overview

- End-to-end machine learning pipeline for **tool wear prediction**
- Reads Excel-based experimental drilling dataset
- Cleans & preprocesses categorical and numeric columns
- Converts labels into binary or multi-class wear categories
- Scales data using `StandardScaler`
- Trains a **Keras Sequential DNN** (with a scikit-learn fallback)
- Provides evaluation, performance metrics, and saved models
- Includes inference script for predicting wear on new data

## 📁 Repository Structure

drilling-tool-wear-dnn/
│── README.md
│── model.py # Training pipeline
│── predict.py # Inference script
│── keras_colab.py # GPU-ready Colab training script
│── data/
│ └── XAI_Drilling_Dataset.xlsx
│── outputs/
│── saved_model/
│── requirements.txt

## 🧠 Model Architecture

Default DNN structure:

- Dense(32, relu)
- Dense(64, relu)
- Dense(32, relu)
- Dense(1, sigmoid)          → binary wear classification  
**or**
- Dense(3, softmax)          → three wear levels (Low/Medium/High)

**Loss:**  
- `binary_crossentropy` or `categorical_crossentropy`

**Optimizer:**  
- `adam`

**Metrics:**  
- Accuracy, Precision, Recall, F1-score

## 📊 Data Preprocessing Steps

1. Load dataset from Excel (`XAI_Drilling_Dataset.xlsx`)
2. Clean missing values
3. Convert non-numeric columns  
   - Material → {K:1, N:2, P:3}  
   - Drill Bit Type → {H:1, N:2, W:3}
4. Convert target labels to encoded numerical classes
5. Scale inputs using `StandardScaler`
6. Split dataset (80% train, 20% test)

## 🚀 Training the Model (Local Machine)

1. Install dependencies:
pip install -r requirements.txt

2. Train model:
python model.py --epochs 200 --batch 16

3. Outputs will be generated in the `outputs/` folder:
- Confusion matrix
- ROC curve (binary)
- Classification report
- Accuracy/loss curves

4. Saved model & scaler stored in `saved_model/`.

## 🌐 Google Colab Training

Run the GPU-supported script:

!python keras_colab.py

This will:
- Load the drilling dataset  
- Preprocess features & target  
- Train the DNN  
- Display evaluation plots  
- Save final `.h5` model and scaler  

## 🔮 Predict Wear for New Samples

Use:

python predict.py --model saved_model/tool_wear_dnn.h5 --scaler saved_model/scaler.joblib --input new_samples.csv

Output saved to:

outputs/predictions.csv

## 🛠 Requirements
pandas
numpy
tensorflow
scikit-learn
matplotlib
seaborn
openpyxl
joblib

## ⭐ Summary

This repository delivers a production-quality machine learning solution for Drilling Tool Wear Prediction with:

- Clean data pipeline  
- Deep Learning model  
- Reusable inference scripts  
- Visual performance reports  
- GPU-ready training flow (Colab)

Perfect for manufacturing research and predictive maintenance projects.

