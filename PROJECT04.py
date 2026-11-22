
# =====================================================
# PROJECT 04: TITANIC SURVIVAL PREDICTION
# Decision Tree Classification using Scikit-Learn
# =====================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    ConfusionMatrixDisplay
)

print("Libraries imported successfully!")

# =====================================================
# 1. LOAD DATASET
# =====================================================

print("\nLoading Titanic Dataset...")

df = sns.load_dataset("titanic")

df = df[["survived", "pclass", "sex", "age", "fare"]]

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# 2. DATA PREPROCESSING
# =====================================================

print("\n========== DATA PREPROCESSING ==========")

# Remove rows with missing values in selected columns
df = df.dropna().copy()

# Convert gender into numerical values
df["sex"] = (df["sex"] == "female").astype(int)

print("\nData After Cleaning:")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)

# =====================================================
# 3. SELECT FEATURES AND TARGET
# =====================================================

X = df.drop(columns=["survived"])
y = df["survived"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================

print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# =====================================================
# 5. BASELINE MODEL
# =====================================================

print("\n========== BASELINE MODEL ==========")

baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("Baseline Accuracy:", round(baseline_accuracy, 3))

# =====================================================
# 6. TRAIN DECISION TREE MODEL
# =====================================================

print("\n========== DECISION TREE MODEL ==========")

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("Decision Tree Model Trained Successfully!")

# =====================================================
# 7. MAKE PREDICTIONS
# =====================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

print("\nActual Values:")
print(y_test.head(10).tolist())

print("\nPredicted Values:")
print(predictions[:10].tolist())

# =====================================================
# 8. MODEL EVALUATION
# =====================================================

print("\n========== MODEL EVALUATION ==========")

model_accuracy = accuracy_score(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    probabilities
)

print("Baseline Accuracy:", round(baseline_accuracy, 3))

print("Decision Tree Accuracy:", round(model_accuracy, 3))

print("Mean Squared Error (Probability):", round(mse, 3))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# =====================================================
# 9. CONFUSION MATRIX
# =====================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, predictions)

print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived", "Survived"]
).plot(cmap="Blues")

plt.title("Titanic Survival - Confusion Matrix")
plt.tight_layout()
plt.savefig("PROJECT04_Confusion_Matrix.png", dpi=150)
plt.show()

# =====================================================
# 10. FEATURE IMPORTANCE
# =====================================================

print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)

plt.title("Decision Tree Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("PROJECT04_Feature_Importance.png", dpi=150)
plt.show()

# =====================================================
# 11. DECISION TREE VISUALIZATION
# =====================================================

plt.figure(figsize=(18, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Titanic Survival Decision Tree")

plt.tight_layout()
plt.savefig("PROJECT04_Decision_Tree.png", dpi=150)
plt.show()

# =====================================================
# 12. FINAL COMPARISON
# =====================================================

print("\n========== MODEL COMPARISON ==========")

print("Baseline Accuracy:", round(baseline_accuracy, 3))
print("Decision Tree Accuracy:", round(model_accuracy, 3))

if model_accuracy > baseline_accuracy:
    print("Decision Tree performed better than the baseline.")
elif model_accuracy == baseline_accuracy:
    print("Decision Tree performed the same as the baseline.")
else:
    print("Baseline performed better than the Decision Tree.")

# =====================================================
# 13. FINAL CONCLUSION
# =====================================================

print("\n========== PROJECT CONCLUSION ==========")

print("1. Titanic dataset was used for survival prediction.")

print("2. Data was divided into training and testing sets.")

print("3. A Decision Tree classifier was trained.")

print("4. The model was evaluated using accuracy,")
print("   confusion matrix and mean squared error.")

print("5. The model was compared with a majority-class baseline.")

print("\nPROJECT 04 COMPLETED SUCCESSFULLY!")

# =====================================================
