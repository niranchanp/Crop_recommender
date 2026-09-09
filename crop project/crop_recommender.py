
"""
Crop Recommender — Multi-Class Classification
Predicts the best crop to grow based on soil nutrients (N, P, K),
temperature, humidity, pH, and rainfall.

Dataset: Crop Recommendation Dataset (Kaggle)
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("Crop_recommendation.csv")

print("Shape:", df.shape)
print("\nFirst rows:\n", df.head())
print("\nClasses:", df['label'].nunique())
print(df['label'].value_counts())

# ---------------------------------------------------------
# 2. QUICK EDA
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['label']).corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 3. PREPROCESS
# ---------------------------------------------------------
X = df.drop(columns=['label'])
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. TRAIN + COMPARE MODELS
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest model: {best_name} ({results[best_name]:.4f} accuracy)")

# ---------------------------------------------------------
# 5. DETAILED EVALUATION OF BEST MODEL
# ---------------------------------------------------------
best_preds = best_model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(y_test, best_preds, target_names=le.classes_))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ---------------------------------------------------------
# 6. FEATURE IMPORTANCE (if Random Forest is best)
# ---------------------------------------------------------
if best_name == "Random Forest":
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    print("\nFeature Importances:\n", importances)

    plt.figure(figsize=(8, 5))
    importances.plot(kind='bar', color='seagreen')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

# ---------------------------------------------------------
# 7. PREDICTION FUNCTION (for demo / submission)
# ---------------------------------------------------------
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Given soil/climate values, return the recommended crop."""
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                             columns=X.columns)
    input_scaled = scaler.transform(input_df)
    pred = best_model.predict(input_scaled)
    return le.inverse_transform(pred)[0]

# Example usage
example = recommend_crop(35, 62, 78, 18.87, 14.0, 7.5, 85.9)
print(f"\nExample recommendation: {example}")

print("\nDone. Saved: correlation_heatmap.png, confusion_matrix.png"
      + (", feature_importance.png" if best_name == "Random Forest" else ""))
