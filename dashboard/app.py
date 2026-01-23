from pathlib import Path
import json
import io

import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, send_file, request

from dashboard.fairness_gaps import compute_gaps


APP_ROOT = Path(__file__).resolve().parent.parent  # project root
REPORTS = APP_ROOT / "outputs" / "reports"

app = Flask(__name__)

# Helpers (load + charts)
def load_summary(name: str) -> dict:
    p = REPORTS / f"{name}_summary.json"
    with open(p, "r") as f:
        return json.load(f)

def load_table(name: str, table: str) -> pd.DataFrame:
    # table in {"by_race", "by_age"}
    p = REPORTS / f"{name}_{table}.csv"
    return pd.read_csv(p)

def make_bar(df, x, y, title):
    fig = px.bar(df, x=x, y=y, title=title, hover_data=["n", "fpr", "fnr"])
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def compute_rates_from_df(df, y_true_col="y_true", y_pred_col="y_pred"):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(df[y_true_col], df[y_pred_col], labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    return float(acc), float(fpr), float(fnr), int(tn), int(fp), int(fn), int(tp)

def subgroup_table(df, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        if len(sub) < 30:
            continue
        acc, fpr, fnr, tn, fp, fn, tp = compute_rates_from_df(sub)
        rows.append(
            {
                "group": g,
                "n": int(len(sub)),
                "acc": acc,
                "fpr": fpr,
                "fnr": fnr,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["group", "n", "acc", "fpr", "fnr", "tn", "fp", "fn", "tp"])
    return pd.DataFrame(rows).sort_values("n", ascending=False)

def gaps_from_subgroup_df(df_sub: pd.DataFrame) -> dict:
    """
    df_sub columns: group, n, acc, fpr, fnr ...
    returns acc_gap, fpr_gap, fnr_gap using max-min
    """
    if df_sub is None or len(df_sub) == 0:
        return {"acc_gap": np.nan, "fpr_gap": np.nan, "fnr_gap": np.nan}

    def _gap(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return np.nan
        return float(s.max() - s.min())

    return {
        "acc_gap": _gap(df_sub["acc"]) if "acc" in df_sub.columns else np.nan,
        "fpr_gap": _gap(df_sub["fpr"]) if "fpr" in df_sub.columns else np.nan,
        "fnr_gap": _gap(df_sub["fnr"]) if "fnr" in df_sub.columns else np.nan,
    }

def pct_reduction(baseline_gaps: dict, mitigated_gaps: dict) -> dict:
    """
    % reduction in gap after mitigation.
    Positive = gap got smaller (good).
    Negative = gap got bigger (bad).
    """
    out = {}
    for k in baseline_gaps:
        b = float(baseline_gaps[k])
        m = float(mitigated_gaps.get(k, np.nan))
        den = max(b, 1e-6)  # prevents divide-by-zero / wild %
        out[k] = 100.0 * (b - m) / den
    return out

def disparity_score(race_gaps: dict, age_gaps: dict, weights=(0.5, 0.5)) -> float:
    """
    Single scalar disparity score.
    Uses (FPR gap + FNR gap) for race and age and combines with weights.
    """
    rw, aw = weights
    race_val = float(race_gaps.get("fpr_gap", 0.0)) + float(race_gaps.get("fnr_gap", 0.0))
    age_val  = float(age_gaps.get("fpr_gap", 0.0)) + float(age_gaps.get("fnr_gap", 0.0))
    return (rw * race_val) + (aw * age_val)

def threshold_sweep(df: pd.DataFrame, score_col="y_score", thresholds=None,
                    objective="min_disparity", min_acc_drop=0.10):
    """
    Finds a threshold that reduces disparity gaps.

    - objective="min_disparity": minimise combined disparity score (race+age FPR/FNR gaps)
    - min_acc_drop: allow up to 10% absolute accuracy drop vs baseline threshold 0.5
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)  # 0.05 steps

    # baseline at 0.5
    df0 = df.copy()
    df0["y_pred"] = (df0[score_col] >= 0.5).astype(int)
    base_acc, base_fpr, base_fnr, tn, fp, fn, tp = compute_rates_from_df(df0)
    base_race = subgroup_table(df0, "race")
    base_age = subgroup_table(df0, "age_group")
    base_race_g = gaps_from_subgroup_df(base_race)
    base_age_g = gaps_from_subgroup_df(base_age)
    base_disp = disparity_score(base_race_g, base_age_g)

    best = None

    for t in thresholds:
        dft = df.copy()
        dft["y_pred"] = (dft[score_col] >= float(t)).astype(int)

        acc, fpr, fnr, tn, fp, fn, tp = compute_rates_from_df(dft)

        # Guard: do not choose a "fair" threshold that destroys performance
        if acc < (base_acc - min_acc_drop):
            continue

        race_sub = subgroup_table(dft, "race")
        age_sub = subgroup_table(dft, "age_group")
        race_g = gaps_from_subgroup_df(race_sub)
        age_g = gaps_from_subgroup_df(age_sub)

        disp = disparity_score(race_g, age_g)

        if best is None or disp < best["disparity"]:
            best = {
                "threshold": float(t),
                "overall": {
                    "acc": float(acc),
                    "fpr": float(fpr),
                    "fnr": float(fnr),
                    "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    "n": int(len(dft)),
                },
                "race_gaps": race_g,
                "age_gaps": age_g,
                "disparity": float(disp),
                "baseline": {
                    "threshold": 0.5,
                    "overall": {"acc": float(base_acc), "fpr": float(base_fpr), "fnr": float(base_fnr)},
                    "race_gaps": base_race_g,
                    "age_gaps": base_age_g,
                    "disparity": float(base_disp),
                },
            }

    return best

# Routes
@app.route("/")
def index():
    baseline = load_summary("baseline_gender_val")
    mitigated = load_summary("fair_arcface_gender_val")
    return render_template("index.html", baseline=baseline, mitigated=mitigated)

@app.route("/compare")
def compare():
    base = load_summary("baseline_gender_val")
    fair = load_summary("fair_arcface_gender_val")

    base_race = load_table("baseline_gender_val", "by_race")
    fair_race = load_table("fair_arcface_gender_val", "by_race")
    base_age = load_table("baseline_gender_val", "by_age")
    fair_age = load_table("fair_arcface_gender_val", "by_age")

    # gaps from the exported reports
    baseline_race_gaps = compute_gaps(REPORTS / "baseline_gender_val_by_race.csv")
    mitigated_race_gaps = compute_gaps(REPORTS / "fair_arcface_gender_val_by_race.csv")
    baseline_age_gaps = compute_gaps(REPORTS / "baseline_gender_val_by_age.csv")
    mitigated_age_gaps = compute_gaps(REPORTS / "fair_arcface_gender_val_by_age.csv")

    race_pct_reduction = pct_reduction(baseline_race_gaps, mitigated_race_gaps)
    age_pct_reduction = pct_reduction(baseline_age_gaps, mitigated_age_gaps)

    # overall disparity scores (single number)
    baseline_disp = disparity_score(baseline_race_gaps, baseline_age_gaps)
    mitigated_disp = disparity_score(mitigated_race_gaps, mitigated_age_gaps)
    disp_reduction_pct = 100.0 * (baseline_disp - mitigated_disp) / max(baseline_disp, 1e-6)

    # Charts
    chart_base_race_acc = make_bar(base_race, "race", "acc", "Baseline Accuracy by Race")
    chart_fair_race_acc = make_bar(fair_race, "race", "acc", "Mitigated Accuracy by Race")
    chart_base_age_acc = make_bar(base_age, "age_group", "acc", "Baseline Accuracy by Age Group")
    chart_fair_age_acc = make_bar(fair_age, "age_group", "acc", "Mitigated Accuracy by Age Group")

    # deltas
    delta_acc = fair["overall"]["acc"] - base["overall"]["acc"]
    delta_fpr = fair["overall"]["fpr"] - base["overall"]["fpr"]
    delta_fnr = fair["overall"]["fnr"] - base["overall"]["fnr"]

    return render_template(
        "compare.html",
        base=base,
        fair=fair,
        delta_acc=delta_acc,
        delta_fpr=delta_fpr,
        delta_fnr=delta_fnr,
        chart_base_race_acc=chart_base_race_acc,
        chart_fair_race_acc=chart_fair_race_acc,
        chart_base_age_acc=chart_base_age_acc,
        chart_fair_age_acc=chart_fair_age_acc,
        baseline_race_gaps=baseline_race_gaps,
        mitigated_race_gaps=mitigated_race_gaps,
        race_pct_reduction=race_pct_reduction,
        baseline_age_gaps=baseline_age_gaps,
        mitigated_age_gaps=mitigated_age_gaps,
        age_pct_reduction=age_pct_reduction,
        baseline_disp=baseline_disp,
        mitigated_disp=mitigated_disp,
        disp_reduction_pct=disp_reduction_pct,
    )

@app.route("/download/<path:filename>")
def download(filename):
    return send_file(REPORTS / filename, as_attachment=True)

PLUGIN_LAST_REPORT = None

@app.route("/download_sample_csv")
def download_sample_csv():
    """
    sample CSV includes y_score so that plugin mitigation can demonstrate threshold tuning
    """
    p = APP_ROOT / "outputs" / "fairface_labels_val.csv"
    df = pd.read_csv(p)
    df = df[df["face_detected"] == 1].copy()
    sample = df.sample(500, random_state=42)

    y_true = sample["gender_id"].astype(int).to_numpy()

    # Creates a synthetic score correlated with truth, with noise
    rng = np.random.RandomState(42)
    score = 0.15 + 0.7 * y_true + 0.15 * rng.randn(len(y_true))
    score = np.clip(score, 0.0, 1.0)

    # baseline y_pred at 0.5
    y_pred = (score >= 0.5).astype(int)

    out = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": score,
            "race": sample["race"].to_numpy(),
            "age_group": sample["age_group"].to_numpy(),
        }
    )

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_plugin_input.csv",
    )

@app.route("/plugin", methods=["GET", "POST"])
def plugin():
    global PLUGIN_LAST_REPORT
    result = None
    mitigated = None
    chart_race_acc = ""
    chart_age_acc = ""

    if request.method == "POST":
        f = request.files.get("file")
        if not f:
            return render_template("plugin.html", result=None, mitigated=None,
                                   chart_race_acc="", chart_age_acc="")

        df = pd.read_csv(f)

        # Required base columns
        required_base = {"y_true", "race", "age_group"}
        missing_base = required_base - set(df.columns)
        if missing_base:
            return render_template(
                "plugin.html",
                result={"error": f"Missing columns: {', '.join(sorted(missing_base))}"},
                mitigated=None,
                chart_race_acc="",
                chart_age_acc="",
            )

        has_pred = "y_pred" in df.columns
        has_score = "y_score" in df.columns

        if not (has_pred or has_score):
            return render_template(
                "plugin.html",
                result={"error": "CSV must include either y_pred (0/1) or y_score (0–1)."},
                mitigated=None,
                chart_race_acc="",
                chart_age_acc="",
            )

        df = df.dropna(subset=["y_true", "race", "age_group"]).copy()
        df["y_true"] = df["y_true"].astype(int)

        # If y_score exists but y_pred doesn't, creates baseline y_pred at 0.5
        if has_score and not has_pred:
            df = df.dropna(subset=["y_score"]).copy()
            df["y_score"] = df["y_score"].astype(float)
            df["y_pred"] = (df["y_score"] >= 0.5).astype(int)
            has_pred = True

        if has_pred:
            df = df.dropna(subset=["y_pred"]).copy()
            df["y_pred"] = df["y_pred"].astype(int)

        overall_acc, overall_fpr, overall_fnr, tn, fp, fn, tp = compute_rates_from_df(df)

        by_race = subgroup_table(df, "race")
        by_age = subgroup_table(df, "age_group")

        # Charts
        if len(by_race) > 0:
            fig_r = px.bar(by_race, x="group", y="acc", title="Audit Accuracy by Race",
                           hover_data=["n", "fpr", "fnr"])
            fig_r.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
            chart_race_acc = fig_r.to_html(full_html=False, include_plotlyjs="cdn")

        if len(by_age) > 0:
            fig_a = px.bar(by_age, x="group", y="acc", title="Audit Accuracy by Age Group",
                           hover_data=["n", "fpr", "fnr"])
            fig_a.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
            chart_age_acc = fig_a.to_html(full_html=False, include_plotlyjs="cdn")

        # Baseline results
        result = {
            "overall": {
                "acc": overall_acc,
                "fpr": overall_fpr,
                "fnr": overall_fnr,
                "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                "n": int(len(df)),
            },
            "race_gaps": gaps_from_subgroup_df(by_race),
            "age_gaps": gaps_from_subgroup_df(by_age),
        }

        # If y_score exists, mitigates via threshold sweep
        if has_score:
            best = threshold_sweep(df, score_col="y_score", objective="min_disparity", min_acc_drop=0.10)
            if best is not None:
                baseline_score = float(best["baseline"]["disparity"])
                mitigated_score = float(best["disparity"])
                disparity_reduction_pct = 100.0 * (baseline_score - mitigated_score) / max(baseline_score, 1e-6)

                mitigated = {
                    "threshold": best["threshold"],
                    "overall": best["overall"],
                    "race_gaps": best["race_gaps"],
                    "age_gaps": best["age_gaps"],
                    "baseline_disparity_score": baseline_score,
                    "mitigated_disparity_score": mitigated_score,
                    "disparity_reduction_pct": disparity_reduction_pct,
                }

                # stores downloadable report
                PLUGIN_LAST_REPORT = {
                    "baseline_overall": result["overall"],
                    "baseline_gaps": {"race": result["race_gaps"], "age": result["age_gaps"]},
                    "mitigated_overall": best["overall"],
                    "mitigated_gaps": {"race": best["race_gaps"], "age": best["age_gaps"]},
                    "chosen_threshold": best["threshold"],
                    "baseline_disparity_score": baseline_score,
                    "mitigated_disparity_score": mitigated_score,
                    "disparity_reduction_pct": disparity_reduction_pct,
                    "schema": {
                        "required_cols": ["y_true", "race", "age_group"],
                        "optional_cols": ["y_pred", "y_score"],
                    },
                }

                # stores mitigated CSV
                df_m = df.copy()
                df_m["y_pred_mitigated"] = (df_m["y_score"] >= best["threshold"]).astype(int)
                buf = io.StringIO()
                df_m[["y_true", "y_pred", "y_pred_mitigated", "y_score", "race", "age_group"]].to_csv(buf, index=False)
                app.config["PLUGIN_LAST_MITIGATED_CSV"] = buf.getvalue().encode("utf-8")
            else:
                mitigated = {"error": "No valid threshold found (constraints too strict)."}
        else:
            # no y_score means no mitigation possible, but still saves baseline audit
            PLUGIN_LAST_REPORT = {
                "baseline_overall": result["overall"],
                "baseline_gaps": {"race": result["race_gaps"], "age": result["age_gaps"]},
                "schema": {
                    "required_cols": ["y_true", "race", "age_group", "y_pred"],
                    "optional_cols": ["y_score"],
                },
            }

    return render_template(
        "plugin.html",
        result=result,
        mitigated=mitigated,
        chart_race_acc=chart_race_acc,
        chart_age_acc=chart_age_acc,
    )

@app.route("/download_plugin_report")
def download_plugin_report():
    global PLUGIN_LAST_REPORT
    if PLUGIN_LAST_REPORT is None:
        return "No plugin report available yet. Run an audit first.", 400

    payload = json.dumps(PLUGIN_LAST_REPORT, indent=2).encode("utf-8")
    return send_file(
        io.BytesIO(payload),
        mimetype="application/json",
        as_attachment=True,
        download_name="audit_report.json",
    )

@app.route("/download_mitigated_csv")
def download_mitigated_csv():
    data = app.config.get("PLUGIN_LAST_MITIGATED_CSV")
    if not data:
        return "No mitigated CSV available yet. Upload a CSV with y_score first.", 400

    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="mitigated_predictions.csv",
    )

if __name__ == "__main__":
    app.run(debug=True)
