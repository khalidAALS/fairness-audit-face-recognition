import pandas as pd


def _gap(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return float("nan")
    return float(series.max() - series.min())


def compute_gaps(by_group_csv_path) -> dict:
    """
    Expects CSV columns: group (or race/age_group), n, acc, fpr, fnr
    Returns gaps (max - min) for acc/fpr/fnr.
    """
    df = pd.read_csv(by_group_csv_path)

    # Some CSVs use 'race' or 'age_group' instead of 'group'
    if "group" not in df.columns:
        for alt in ["race", "age_group"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "group"})
                break

    return {
        "acc_gap": _gap(df["acc"]) if "acc" in df.columns else float("nan"),
        "fpr_gap": _gap(df["fpr"]) if "fpr" in df.columns else float("nan"),
        "fnr_gap": _gap(df["fnr"]) if "fnr" in df.columns else float("nan"),
    }
