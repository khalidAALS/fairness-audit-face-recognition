import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

OUT_DIR = Path("outputs/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_rates(y_true, y_pred):
    # Binary labels assumed: 0/1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    return float(acc), float(fpr), float(fnr), int(tn), int(fp), int(fn), int(tp)

def subgroup_report(df, y_true_col, y_pred_col, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        if len(sub) < 30:
            continue
        acc, fpr, fnr, tn, fp, fn, tp = compute_rates(sub[y_true_col].values, sub[y_pred_col].values)
        rows.append({
            group_col: g,
            "n": int(len(sub)),
            "acc": acc,
            "fpr": fpr,
            "fnr": fnr,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        })
    return pd.DataFrame(rows).sort_values(["n"], ascending=False)

def run_experiment(name, emb_train_path, labels_train_csv, emb_val_path, labels_val_csv, label_col="gender_id"):
    print(f"\n=== Exporting metrics for: {name} ===")
    Xtr = np.load(emb_train_path)
    Xva = np.load(emb_val_path)

    dtr = pd.read_csv(labels_train_csv)
    dva = pd.read_csv(labels_val_csv)

    # Uses only detected faces for training/eval
    dtr = dtr[dtr["face_detected"] == 1].copy()
    dva = dva[dva["face_detected"] == 1].copy()

    # Aligns arrays with filtered rows
    tr_idx = dtr.index.values
    va_idx = dva.index.values
    Xtr = Xtr[tr_idx]
    Xva = Xva[va_idx]

    ytr = dtr[label_col].astype(int).values
    yva = dva[label_col].astype(int).values

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xva)

    dva = dva.copy()
    dva["y_true"] = yva
    dva["y_pred"] = yhat

    overall_acc, overall_fpr, overall_fnr, tn, fp, fn, tp = compute_rates(dva["y_true"].values, dva["y_pred"].values)

    by_race = subgroup_report(dva, "y_true", "y_pred", "race")
    by_age = subgroup_report(dva, "y_true", "y_pred", "age_group")

    # Saves CSVs for dashboard charts
    by_race_path = OUT_DIR / f"{name}_by_race.csv"
    by_age_path = OUT_DIR / f"{name}_by_age.csv"
    by_race.to_csv(by_race_path, index=False)
    by_age.to_csv(by_age_path, index=False)

    # Saves a compact JSON summary
    summary = {
        "name": name,
        "n_val_detected": int(len(dva)),
        "overall": {
            "acc": overall_acc,
            "fpr": overall_fpr,
            "fnr": overall_fnr,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        },
        "files": {
            "by_race_csv": str(by_race_path),
            "by_age_csv": str(by_age_path),
        }
    }

    json_path = OUT_DIR / f"{name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", json_path)
    print("Saved:", by_race_path)
    print("Saved:", by_age_path)
    print("Overall acc/fpr/fnr:", overall_acc, overall_fpr, overall_fnr)

def main():
    # Baseline (FaceNet embeddings)
    run_experiment(
        name="baseline_gender_val",
        emb_train_path="outputs/fairface_embeddings_train.npy",
        labels_train_csv="outputs/fairface_labels_train.csv",
        emb_val_path="outputs/fairface_embeddings_val.npy",
        labels_val_csv="outputs/fairface_labels_val.csv",
        label_col="gender_id",
    )

    # Mitigated (Fair ArcFace backbone embeddings)
    run_experiment(
        name="fair_arcface_gender_val",
        emb_train_path="outputs/fair_arcface_embeddings_train.npy",
        labels_train_csv="outputs/fair_arcface_labels_train.csv",
        emb_val_path="outputs/fair_arcface_embeddings_val.npy",
        labels_val_csv="outputs/fair_arcface_labels_val.csv",
        label_col="gender_id",
    )

if __name__ == "__main__":
    main()
