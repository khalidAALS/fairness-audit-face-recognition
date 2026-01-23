import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train = np.load("outputs/fair_arcface_embeddings_train.npy")
y_train_df = pd.read_csv("outputs/fair_arcface_labels_train.csv")

X_val = np.load("outputs/fair_arcface_embeddings_val.npy")
y_val_df = pd.read_csv("outputs/fair_arcface_labels_val.csv")

train_mask = y_train_df["face_detected"] == 1
val_mask = y_val_df["face_detected"] == 1

Xtr = X_train[train_mask.values]
Xva = X_val[val_mask.values]

ytr = y_train_df.loc[train_mask, "gender_id"].values
yva = y_val_df.loc[val_mask, "gender_id"].values

race_va = y_val_df.loc[val_mask, "race"].values
age_va = y_val_df.loc[val_mask, "age_group"].values

clf = LogisticRegression(max_iter=4000)
clf.fit(Xtr, ytr)

pred = clf.predict(Xva)

overall_acc = accuracy_score(yva, pred)
print("Overall gender accuracy (val) [Fair ArcFace-backbone embeddings]:", overall_acc)

def subgroup_report(name, groups):
    print(f"\n=== Subgroup report by {name} ===")
    for g in sorted(set(groups)):
        idx = groups == g
        acc = accuracy_score(yva[idx], pred[idx])
        tn, fp, fn, tp = confusion_matrix(yva[idx], pred[idx], labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"{g:15s}  n={idx.sum():5d}  acc={acc:.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}")

subgroup_report("race", race_va)
subgroup_report("age_group", age_va)
