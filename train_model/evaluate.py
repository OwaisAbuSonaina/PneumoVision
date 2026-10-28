import glob
import numpy as np
from sklearn.metrics import (
    roc_curve,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import pandas as pd
from tensorflow.keras.models import load_model

from preprocess import load_and_preprocess, X_val, y_val


trained_model = load_model("resnet_model.h5")

val_probs = trained_model.predict(X_val).ravel()

######################################
# Threshold tuning
######################################
fpr, tpr, thresholds = roc_curve(y_val, val_probs)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_thresh = thresholds[best_idx]
print("*"*30)
print(f"Optimal sigmoid threshold = {best_thresh:.3f}")
print("*"*30)

######################################
# Import test data
######################################
test_n = glob.glob("./chest_xray/test/NORMAL/*.jpeg")
test_p = glob.glob("./chest_xray/test/PNEUMONIA/*.jpeg")
X_test_n, y_test_n = load_and_preprocess(test_n, label=0)
X_test_p, y_test_p = load_and_preprocess(test_p, label=1)

X_test = np.concatenate([X_test_n, X_test_p], axis=0)
y_test = np.concatenate([y_test_n, y_test_p], axis=0)

######################################
# Evaluate
######################################
test_probs = trained_model.predict(X_test).ravel()

y_pred = (test_probs >= best_thresh).astype(int)  # best_thresh

loss, acc, auc = trained_model.evaluate(X_test, y_test, verbose=0)
print("*"*30)
print(f"Test Loss: {loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")


print("*"*30)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Pneumonia"]))



cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["Normal (true)", "Pneumonia (true)"],
    columns=["Normal (pred)", "Pneumonia (pred)"],
)
print("*"*30)
print("Confusion Matrix:")
print(cm_df)

print("*"*30)
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
