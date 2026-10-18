import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt


# -----------------------------
# 1. LOAD DATASET
# -----------------------------
train = pd.read_csv("train.txt", header=None)
test = pd.read_csv("test.txt", header=None)

y_train = (train[41] != "normal").astype(int)
y_test = (test[41] != "normal").astype(int)
train = train.drop(columns=[41, 42])
test = test.drop(columns=[41, 42])

# -----------------------------
# 2. FORCE ENCODE ALL COLUMNS
# -----------------------------
for col in train.columns:
    le = LabelEncoder()
    
    combined = pd.concat([train[col], test[col]])
    le.fit(combined.astype(str))
    
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# -----------------------------
# 3. SPLIT FEATURES & LABELS
# -----------------------------
X_train = train
X_test = test

# -----------------------------
# 4. SCALE DATA
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
model = IsolationForest(n_estimators=200, contamination=0.3, random_state=42)
model.fit(X_train[y_train == 0])

# -----------------------------
# 6. PREDICT
# -----------------------------
pred = model.predict(X_test)

# Convert: -1 = anomaly → 1, 1 = normal → 0
pred = [1 if p == -1 else 0 for p in pred]

# -----------------------------
# 7. ACCURACY, PRECISION & RECALL
# -----------------------------
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# -----------------------------
# 8. GRAPH
# -----------------------------
plt.figure()
plt.scatter(range(len(pred)), pred)
plt.title("Anomaly Detection Output")
plt.xlabel("Samples")
plt.ylabel("Prediction (0=Normal, 1=Attack)")
plt.show()

import joblib

joblib.dump(model, "ids_model.pkl")
joblib.dump(scaler, "scaler.pkl")