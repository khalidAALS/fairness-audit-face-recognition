# Fairness Audit Dashboard and Mitigation Plugin

## Paper
Pending

## Overview

Fair-Face is a Flask-based dashboard and plugin designed to audit classification fairness across demographic groups (race and age), and apply simple mitigation strategies to reduce disparities in error rates.

The system evaluates fairness using standard metrics such as False Positive Rate (FPR) and False Negative Rate (FNR), and provides threshold-based mitigation to reduce group-level disparities.

## Live Demo

Accessible at:  
https://fair-face-dashboard-v1.salmonsand-59d31c0b.italynorth.azurecontainerapps.io/

---

## Features

### Dashboard
- Baseline vs mitigation model comparison  
- Group metrics by race and age  
- Fairness gap reporting (max–min across groups)  
- Interactive visualisation (Plotly charts)  

---

### Plugin Audit (CSV Upload)

Upload results from any classifier or face recognition pipeline to compute:

- Overall accuracy, FPR, FNR  
- Disparity gaps by race and age  
- Optional threshold tuning and mitigation (if `y_score` is provided)  

Downloadable outputs:
- `audit_report.json`  
- `mitigation_predictions.csv` (if mitigation is applied)  

---

## CSV Input Format

Required columns:
- `y_true` (0/1)  
- `race` (string group label)  
- `age_group` (string group label)  

Plus one of:
- `y_pred` (0/1)  
- OR `y_score` (continuous score)  

---

## Notes

- If only `y_score` is provided, predictions are generated using a default threshold of 0.5  
- If `y_score` is available, the system can optimise thresholds to reduce disparity  

Sample file available at:
`/download_sample_csv`

---

## Fairness Metrics

For each group (race and age), the system computes:

- Accuracy  
- False Positive Rate (FPR)  
- False Negative Rate (FNR)  

### Disparity Calculation

```
Gap(metric) = max(metric across groups) - min(metric across groups)
```

Reported as:
- Race FPR gap  
- Race FNR gap  
- Age FPR gap  
- Age FNR gap  

---

## Mitigation Method

If `y_score` is available, the system performs threshold sweeping (default: 0.05 → 0.95) and selects the threshold that:

- Minimises disparity (FPR/FNR gaps)  
- Limits performance degradation (accuracy constraint)  

### Combined Disparity Score

```
disparity = (race_fpr_gap + race_fnr_gap) * 0.5 + (age_fpr_gap + age_fnr_gap) * 0.5
```

---

## Outputs

- Selected threshold  
- Mitigated overall metrics  
- Mitigated disparity gaps  
- Percentage reduction in disparity  

---

## Installation

### Web Dashboard

```
pip install -r requirements.txt
```

### Full Training Environment

```
pip install -r requirements-ml.txt
```

---

## Local Run

Create and activate virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Run the dashboard:

```
python dashboard/app.py
```

Open:
http://127.0.0.1:5000

---

## Deployment

### CI/CD Pipeline

GitHub → Docker Hub

- Builds Docker image (multi-architecture: amd64 + arm64)  
- Pushes to Docker Hub: `khalidaals/fair-face-dashboard:latest`  

### Azure Container Apps

- Pulls Docker image  
- Deploys and exposes via HTTPS  

---

## Project Structure

```
dashboard/
  app.py
  templates/
  static/

src/
  training, evaluation, audit scripts

outputs/
  reports/

requirements.txt
Dockerfile
Procfile
```

---

## Ethical Disclaimer

This project evaluates demographic fairness using dataset-provided labels.  
Demographic labels may be imperfect or socially sensitive, and results should be interpreted carefully.

This tool is intended for auditing and research purposes only, and should not be used as the sole basis for high-stakes or sensitive decisions.
