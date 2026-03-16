# =========================================================
# VANDERBILT ADHD TRAINING PIPELINE
# Research-Grade Version (No Data Leakage)
# =========================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1️⃣ LOAD DATA
# =========================================================
df = pd.read_csv("Evaluation Form (Responses) - Form Responses 1 (1).csv")

# Remove timestamp column if present
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# =========================================================
# 2️⃣ ENCODE RESPONSES
# =========================================================
mapping_4 = {
    "Never": 0,
    "Occasionally": 1,
    "Often": 2,
    "Very Often": 3
}

mapping_5 = {
    "Excellent": 0,
    "Above Average": 1,
    "Average": 2,
    "Somewhat of a Problem": 3,
    "Problematic": 4
}

# First 47 questions
for col in df.columns[:47]:
    df[col] = df[col].map(mapping_4)

# Last 8 questions
for col in df.columns[47:55]:
    df[col] = df[col].map(mapping_5)

df = df.fillna(0)

# =========================================================
# 3️⃣ FEATURE ENGINEERING (CLINICAL SUBSCALES)
# =========================================================
df["Inattention_score"] = df.iloc[:,0:9].sum(axis=1)
df["Hyperactivity_score"] = df.iloc[:,9:18].sum(axis=1)
df["ODD_score"] = df.iloc[:,18:26].sum(axis=1)
df["Conduct_score"] = df.iloc[:,26:40].sum(axis=1)
df["Anxiety_score"] = df.iloc[:,40:47].sum(axis=1)
df["Performance_score"] = df.iloc[:,47:55].sum(axis=1)

# =========================================================
# 4️⃣ CREATE DSM-BASED ADHD LABEL
# =========================================================
def create_label(row):
    inatt = sum(row[0:9] >= 2)
    hyper = sum(row[9:18] >= 2)
    impairment = sum(row[47:55] >= 3)

    if impairment >= 1:
        if inatt >= 6 and hyper >= 6:
            return "Combined"
        elif inatt >= 6:
            return "Inattentive"
        elif hyper >= 6:
            return "Hyperactive"
        else:
            return "Non-ADHD"
    else:
        return "Non-ADHD"

df["ADHD_Type"] = df.apply(create_label, axis=1)

# =========================================================
# 5️⃣ SELECT MODEL FEATURES
# =========================================================
features = [
    "Inattention_score",
    "Hyperactivity_score",
    "ODD_score",
    "Conduct_score",
    "Anxiety_score",
    "Performance_score"
]

X = df[features]
y = df["ADHD_Type"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================================================
# 6️⃣ TRAIN-TEST SPLIT (STRATIFIED)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# =========================================================
# 7️⃣ MODEL (OVERFITTING CONTROLLED)
# =========================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

# =========================================================
# 8️⃣ CROSS VALIDATION (5-FOLD)
# =========================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skf,
    scoring="f1_macro"
)

print("Mean CV Macro F1:", cv_scores.mean())

# =========================================================
# 9️⃣ FINAL TRAINING
# =========================================================
model.fit(X_train, y_train)

# =========================================================
# 🔟 TEST EVALUATION
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================================================
# CONFUSION MATRIX (TEST SET)
# =========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# =========================================================
# ROC-AUC (BINARY ONLY)
# =========================================================
if len(le.classes_) == 2:
    auc = roc_auc_score(y_test, y_prob[:,1])
    print("Test ROC-AUC:", auc)

    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
plt.figure(figsize=(6,5))
importance = model.feature_importances_
plt.barh(features, importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# =========================================================
# SAVE MODEL + METADATA
# =========================================================
joblib.dump(model, "adhd_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(features, "feature_columns.pkl")

print("\nModel training complete and files saved successfully.")