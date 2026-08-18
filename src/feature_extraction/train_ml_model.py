"""
train_ml_model.py - Train a Random Forest to detect deepfakes using FFT features.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the balanced features we just extracted
print("Loading features...")
X = np.load('X_features_balanced.npy')
y = np.load('y_labels_balanced.npy')

print(f"Loaded {len(X)} samples, each with {X.shape[1]} features.")
print(f"Class distribution: Real(0)={np.sum(y==0)}, Fake(1)={np.sum(y==1)}")

# 2. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# 3. Train the Random Forest
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=10,       # Limit depth to avoid overfitting
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# 5. Detailed report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📉 Confusion Matrix:")
print("              Predicted Real  Predicted Fake")
print(f"Actual Real   {cm[0,0]:<12}  {cm[0,1]:<12}")
print(f"Actual Fake   {cm[1,0]:<12}  {cm[1,1]:<12}")

# 7. Feature importance (Which frequencies matter most?)
importances = model.feature_importances_
top_indices = np.argsort(importances)[-5:][::-1]  # Top 5 most important frequencies
print("\n🎯 Top 5 most important frequency bins:")
for idx in top_indices:
    print(f"  Frequency bin {idx}: importance {importances[idx]:.4f}")

print("\n💡 If the top frequencies are around 1000 Hz (bin ~30-40),")
print("   it confirms the model is using the human voice resonance to detect fakes!")