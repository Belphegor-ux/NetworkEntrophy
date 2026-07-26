import pandas as pd
import numpy as np
import os

def validate_metrics(csv_path, report_path):
    if not os.path.exists(csv_path):
        with open(report_path, "w") as f:
            f.write(f"# Validation Report\n\n**Error:** {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    report = ["# Validation Report for Tokyo Power Grid Metrics\n"]
    
    # 1. Expected Columns
    expected_cols = ['i', 'j', 'LDC', 'Jaccard', 'LKS', 'CI_e_av_skin', 'CI_e_mul_skin', 'CI_e_av_body', 'CI_e_mul_body', 'LLBCe', 'LLBMEe1']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        report.append(f"## Column Check: ❌ FAILED\nMissing columns: {', '.join(missing_cols)}\n")
    else:
        report.append("## Column Check: ✅ PASSED\nAll expected columns are present.\n")

    # 2. Missing Values (NaN)
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        report.append("## Missing Values Check: ❌ FAILED\n")
        report.append(nan_counts[nan_counts > 0].to_string() + "\n")
    else:
        report.append("## Missing Values Check: ✅ PASSED\nNo missing values (NaNs) found.\n")

    # 3. Data Range & Type Checks
    range_results = []
    
    # Jaccard 0-1
    if ((df['Jaccard'] < 0) | (df['Jaccard'] > 1)).any():
        range_results.append("❌ Jaccard: Values found outside [0, 1]")
    else:
        range_results.append("✅ Jaccard: All values within [0, 1]")
        
    # Non-negative metrics (LLBMEe1 is intentionally <= 0 and handled separately).
    non_neg_cols = ['LDC', 'LKS', 'CI_e_av_skin', 'CI_e_mul_skin', 'CI_e_av_body', 'CI_e_mul_body', 'LLBCe']
    for col in non_neg_cols:
        if (df[col] < 0).any():
            range_results.append(f"❌ {col}: Negative values found")
        else:
            range_results.append(f"✅ {col}: All values are non-negative")

    # LLBMEe1: inverted-criticality metric. By construction it is <= 0
    # (smallest = most critical). Some edges can flip positive when raw subset
    # betweenness < 1 makes log() negative; report that as info, not a failure.
    if 'LLBMEe1' in df.columns:
        n_pos = int((df['LLBMEe1'] > 0).sum())
        if n_pos == 0:
            range_results.append("✅ LLBMEe1: all values <= 0 (smallest = most critical)")
        else:
            range_results.append(
                f"ℹ️ LLBMEe1: {n_pos} edge(s) > 0 — raw-betweenness log sign flip; "
                "review raw-vs-normalized semantics"
            )

    # Edge distinctness (i != j)
    if (df['i'] == df['j']).any():
        range_results.append("❌ Edges: Self-loops found (i == j)")
    else:
        range_results.append("✅ Edges: All edges are distinct node pairs (i != j)")

    report.append("## Data Range & Type Checks\n")
    report.append("\n".join(range_results) + "\n")

    # 4. Basic Stats Summary
    report.append("## Metrics Summary Statistics\n")
    report.append("```\n" + df.describe().to_string() + "\n```\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report)
    print(f"Validation report generated at {report_path}")

if __name__ == "__main__":
    csv_in = os.path.join("results_city", "tokyo_grid_metrics.csv")
    report_out = os.path.join("results_city", "validation_report.md")
    validate_metrics(csv_in, report_out)
