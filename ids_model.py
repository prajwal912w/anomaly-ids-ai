import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# -----------------------------
# 1. LOAD DATASET
# -----------------------------
train = pd.read_csv("train.txt", header=None)
test = pd.read_csv("test.txt", header=None)

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
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# -----------------------------
# 4. SCALE DATA
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# -----------------------------
# 6. PREDICT
# -----------------------------
pred = model.predict(X_test)

# Convert: -1 = anomaly → 1, 1 = normal → 0
pred = [1 if p == -1 else 0 for p in pred]

# -----------------------------
# 7. ACCURACY
# -----------------------------
# Convert y_test to binary (just in case)
y_test = [0 if y == y_test[0] else 1 for y in y_test]

accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

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