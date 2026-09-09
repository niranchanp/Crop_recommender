Edited crop_recommender.py
Ran command: ` c:; cd 'c:\Users\Rana T\OneDrive\Documents\crop project'; & 'c:\Users\Rana T\AppData\Local\Programs\Python\Python313\python.exe' 'c:\Users\Rana T\.antigravity-ide\extensions\ms-python.debugpy-2026.6.0-win32-x64\bundled\libs\debugpy\launcher' '57480' '--' 'C:\Users\Rana T\OneDrive\Documents\crop project\crop_recommender.py' `
Viewed crop_recommender.py:115-128

Here is a complete, structured project description for your **Crop Recommendation System**:

---

# 🌾 Smart Agriculture: Crop Recommendation System

## 📌 Project Overview
The **Crop Recommendation System** is a machine learning model built to optimize agricultural decision-making by predicting the most suitable crop to cultivate. By analyzing specific soil nutrient levels and regional environmental conditions, the system assists farmers and agricultural experts in maximizing crop yield and resource efficiency.

---

## 🎯 Key Objectives
1. **Multi-Class Classification**: Predict the optimal crop from **22 distinct crop types** (e.g., *Rice, Maize, Chickpea, Cotton, Coffee, Apple, Banana, Grapes, Watermelon*, etc.).
2. **Model Benchmarking**: Train and evaluate 4 different machine learning models to identify the highest performing classifier.
3. **Inference Function**: Provide an easy-to-use recommendation function (`recommend_crop`) that predicts the best crop for custom real-world soil/weather parameters.
4. **Data Visualization**: Generate analytical charts (Correlation Heatmap, Confusion Matrix, and Feature Importance).

---

## 📊 Dataset & Features
The project uses the **Crop Recommendation Dataset** containing **2,200 samples** (100 samples per crop class).

### Input Features (7 Parameters):
- **`N`**: Nitrogen content ratio in soil
- **`P`**: Phosphorus content ratio in soil
- **`K`**: Potassium content ratio in soil
- **`temperature`**: Ambient temperature in °C
- **`humidity`**: Relative humidity percentage (%)
- **`ph`**: Soil pH value (acidic/alkaline balance)
- **`rainfall`**: Average rainfall in mm

### Target Variable:
- **`label`**: Categorical target representing 1 of 22 crop types.

---

## ⚙️ Workflow & Architecture

```
Raw Dataset (2,200 rows)
       │
       ▼
Preprocessing (Label Encoding + Stratified 80/20 Split + StandardScaler)
       │
       ▼
Model Training & Comparison (Logistic Regression, Decision Tree, KNN, Random Forest)
       │
       ▼
Evaluation (Accuracy: 99.55% | Confusion Matrix & Feature Importance)
       │
       ▼
Inference API (`recommend_crop(N, P, K, temp, humidity, ph, rainfall)`)
```

1. **Preprocessing**:
   - **Label Encoding**: Converts target crop names into numerical indices (`0`–`21`).
   - **Stratified Split**: Ensures balanced representation across all 22 crops in train (80%) and test (20%) sets.
   - **Standardization (`StandardScaler`)**: Normalizes input values to zero mean and unit variance.

2. **Model Evaluation Results**:
   | Model | Accuracy |
   | :--- | :--- |
   | **Random Forest Classifier** | **99.55%** ⭐ (Best Model) |
   | **Decision Tree Classifier** | 97.95% |
   | **K-Nearest Neighbors (KNN)** | 97.95% |
   | **Logistic Regression** | 97.27% |

3. **Key Feature Importances**:
   - **Rainfall** (21.96%) and **Humidity** (21.71%) are the most critical factors influencing crop selection, followed by **Potassium (K)** (18.08%) and **Phosphorus (P)** (15.13%).

---

## 🚀 Usage Example
```python
# Function call with custom soil/climate inputs
recommended = recommend_crop(N=35, P=62, K=78, temperature=18.87, humidity=14.0, ph=7.5, rainfall=85.9)
print(recommended) # Output: 'chickpea'
```

---

## 📂 Output Artifacts Generated
- `correlation_heatmap.png`: Displays pairwise correlations between soil and climate variables.
- `confusion_matrix.png`: Shows classification accuracy across all 22 crop classes.
- `feature_importance.png`: Visualizes feature impact scores for the Random Forest model.
