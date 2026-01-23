**##fair-face: Fairness Audit Dashboard and Mitigation Plugin.**
a Flask dashboard that sudits classification fairness across demographic groups (race and age) and provides simple mitigation methods (threshold tuning) to reduce disparity in error rates (FPR and FNR gaps).

**accessible at**: https://fair-face-dashboard-v1.salmonsand-59d31c0b.italynorth.azurecontainerapps.io/

**##Features**
**#Dashboard**
- Baseline vs mitigation model comparison
- Group metrics by race and age groups
- Fairness gap reporting (max-min across groups)
- Visulisation (Ploty charts)

**#Plugin Audit (CSV upload)**
Upload results from any classifier / face recognition pipeline to compute:
- Overall accuracy, FPR, FNR.
- Disparity gaps by race and age.
- Optional thresholf tuning and mitigation if 'y_score' is included.
- Downloadable:
    'Audit_report.json'.
    'mitigation_predications.csv' #if mitigation is ran.

**#CSV Input Format (for plugin)**
Required columns:
    'y_true' (0/1)
    'race' (string group label)
    'age_group' (string group label)
plus either:
    'y_pred' (0/1) or 'y_score' (0/1)

**#Note**
If only 'y_score' is provided then the baseline predictions are created at threshold 0.5.
If 'y_score' is procided then the plugin can search threshold to reduce demographic disparity.

**#Download a sample file for the app:**
    '/download_sample_csv'

**#How Fairness is Measured**
For each group (race and age) the system computes
- Accuracy
- False Positive Rate (FPR)
- False Negative Rate (FNR)

**#Disparity gaps are computed as**
> Gap(metrics) = max(metric across groups) - min(metric across groups)
Reported as:
- Race FPR gap and Race FNR gap.
- Age FPR gap and Age FNR gap.

**#A combined disparity score is used during mitigation as:**
- 'disparity = (race_fpr_gap + race_fnr_gap) * 0.5 + (age_fpr_gap + age_fnr_gap) * 0.5'

**#Mitigation Method**
If 'y_score' is available, the system will sweep the thresholds (defailt: 0.05 -> 0.95) and select the threshold that:
- Minimises the disparity score based on FPR/FNR gap.
- While limiting performance degradation (Accuracy drop constraint).

**#Outputs Include**
- Choden threshold.
- Mitigated overall metrics.
- Mitigated gaps.
- Disparity reduction %.

**Web and Dashboard install:** pip install -r requirements.txt
**full training install:** pip install -r requirements-ml.txt

**#local run**
create and activate the cirtual enviroment: 
'''bash
python3 -m venv .venv
source .venv/bin/activate

run the dashboard: python dashboard/app/py
open: http://127.0.0.1:5000

**#Deployment CI/CD**
GitHub -> Docker Hub (CI)
This repo includes a GutHub actions workflow that inludes:
- Builds a docker image (Multi arch: amd64 and arm64)
- Pushed it to docker hub: khalidaals/fair-face-dashboard:latest

**#Azure container apps**
Azure container apps pulls the docker hub image and exposes it via HTTPS.

**#Project structure**
dashboard/
    app.py
    templates/
    static/
src/
    training + evaluating + audit scripts
outputs/
    reports/ #kept in git as dashboard artifacts
    fairface_labels_val.csv
requirements.txt
Dockerfile
procfile

**#Ethical Disclaimer**
This project evaluates demographic fairness using datasets provided lavels and demographic labels may be imperfect or socially sensitive and metrics should be interpreted carefully. This tool is intended for aduditing and research puposes and should not be used as a sole basis for sensitive and high stake decisions

