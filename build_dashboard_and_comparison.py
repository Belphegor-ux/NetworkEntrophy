"""
Build the unified dashboard data + static-vs-iterative comparison artifacts.

Consumes run_network_suite.py outputs (results_<name>/) so every network shares
the same canonical metric schema (LDC, Jaccard, LKS, 4x CI, LLBCe, LLBMEe1).

For each network:
  - STATIC curves are recomputed here (cheap: O(m) per metric, no iterative recompute)
    from the input edge-list, using the shared NetworkDismantler.
  - ITERATIVE curves are read from results_<name>/<name>_iter_<metric>_rgc.csv
    (already produced by the suite).

Outputs:
  dashboard_app/dashboard_data.js   - `const chartData = {...};`  (consumed by dashboard.js)
  dashboard_app/dashboard_data.json - same payload as JSON
  static_vs_iterative_comparison.xlsx - per-network + aggregate + cross-network sheets
  results_comparison/static_vs_iterative_auc.csv      - tidy long table
  results_comparison/static_vs_iterative_scatter.png  - static vs iterative AUC scatter
  results_comparison/auc_by_metric.png                - mean AUC per metric, static vs iterative

Run AFTER run_all_conference.sh and run_jazz_tokyo_suite.sh finish.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "PowerGrid_City"))

from network_utils import NetworkDismantler, _resolve_reverse  # noqa: E402
from run_network_suite import RANK_GROUPS, load_lcc_graph  # noqa: E402

# (display name, suite --name, input edge-list CSV). Order = dashboard dropdown order.
NETWORKS = [
    ("Karate", "karate_check", "Networks to check/karate.csv"),
    ("Football", "football_new", "Networks to check/football.csv"),
    ("Jazz", "jazz", "_derived_jazz.csv"),
    ("Tokyo City", "tokyo", "_derived_tokyo.csv"),
    ("Dolphins", "dolphins", "Networks to check/dolphins.csv"),
    ("Les Miserables", "lesmis", "Networks to check/lesmis_edges.csv"),
    ("Baseball Steroids", "baseball", "Networks to check/Baseball steroid player_edges.csv"),
    ("Transport", "transport", "Networks to check/2transport_edges_re.csv"),
    ("C. elegans Gap Junction", "hermaphrodite", "Networks to check/hermaphrodite_gap_junction.csv"),
]

# Canonical metric column order for tables/plots.
METRIC_ORDER = [
    "LDC", "Jaccard", "LKS",
    "CI_e_av_skin", "CI_e_mul_skin", "CI_e_av_body", "CI_e_mul_body",
    "LLBCe", "LLBMEe1",
]


def _auc(y: list[float]) -> float:
    x = np.linspace(0, 1, len(y))
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def static_curves(G: nx.Graph) -> dict:
    """Recompute static RGC curves for every metric column."""
    dism = NetworkDismantler(G)
    out: dict = {}
    for _label, func, reverse in RANK_GROUPS:
        df = func(G)
        for col in [c for c in df.columns if c not in ("i", "j")]:
            rev = _resolve_reverse(reverse, col)
            rgc = dism.get_static_curve(df, col, reverse=rev)
            x = np.linspace(0, 1, len(rgc)).tolist()
            out[col] = {"x": x, "y": rgc, "auc": _auc(rgc)}
    return out


def iterative_curves(suite_name: str, out_dir: Path) -> dict:
    """Read iterative RGC curves saved by run_network_suite."""
    out: dict = {}
    for col in METRIC_ORDER:
        csv = out_dir / f"{suite_name}_iter_{col}_rgc.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        out[col] = {
            "x": df["Fraction_Removed"].tolist(),
            "y": df["RGC"].tolist(),
            "auc": _auc(df["RGC"].tolist()),
        }
    return out


def build() -> dict:
    chart_data: dict = {}
    for display, suite_name, csv_rel in NETWORKS:
        csv_path = ROOT / csv_rel
        out_dir = ROOT / f"results_{suite_name}"
        if not csv_path.exists() or not out_dir.exists():
            print(f"  SKIP {display}: missing {csv_path if not csv_path.exists() else out_dir}")
            continue
        G = load_lcc_graph(str(csv_path))
        stat = static_curves(G)
        itr = iterative_curves(suite_name, out_dir)
        chart_data[display] = {"Static": stat, "Iterative": itr}
        print(f"  {display:26s} static={len(stat)} iter={len(itr)} "
              f"(LCC {G.number_of_nodes()}n/{G.number_of_edges()}e)")
    return chart_data


def write_dashboard(chart_data: dict) -> None:
    dash = ROOT / "dashboard_app"
    dash.mkdir(exist_ok=True)
    with open(dash / "dashboard_data.json", "w") as f:
        json.dump(chart_data, f)
    with open(dash / "dashboard_data.js", "w") as f:
        f.write("const chartData = ")
        json.dump(chart_data, f)
        f.write(";")
    print(f"Wrote dashboard_app/dashboard_data.js (+ .json) - {len(chart_data)} networks")


def comparison_table(chart_data: dict) -> pd.DataFrame:
    rows = []
    for ds, modes in chart_data.items():
        for m in METRIC_ORDER:
            s = modes.get("Static", {}).get(m)
            i = modes.get("Iterative", {}).get(m)
            if s is None or i is None:
                continue
            sa, ia = s["auc"], i["auc"]
            rows.append({
                "Network": ds, "Metric": m,
                "Static_AUC": sa, "Iterative_AUC": ia,
                "Diff_I_minus_S": ia - sa,
                "Pct_Change": (ia - sa) / sa if sa else 0.0,
                "Better": "Iterative" if ia < sa else "Static",
            })
    return pd.DataFrame(rows)


def write_xlsx(df: pd.DataFrame) -> None:
    xlsx = ROOT / "static_vs_iterative_comparison.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Static vs Iterative", index=False)
        for ds in df["Network"].unique():
            sub = df[df["Network"] == ds]
            sheet = ds[:31]
            sub.to_excel(xl, sheet_name=sheet, index=False)
        # Cross-network summary: iterative win-rate per metric + mean delta
        summ = (
            df.groupby("Metric")
            .agg(
                mean_static=("Static_AUC", "mean"),
                mean_iterative=("Iterative_AUC", "mean"),
                mean_delta=("Diff_I_minus_S", "mean"),
                iterative_wins=("Better", lambda s: int((s == "Iterative").sum())),
                n=("Better", "size"),
            )
            .reindex([m for m in METRIC_ORDER if m in set(df["Metric"])])
            .reset_index()
        )
        summ["iterative_win_rate"] = summ["iterative_wins"].astype(str) + "/" + summ["n"].astype(str)
        summ.to_excel(xl, sheet_name="Cross-Network Summary", index=False)
    print(f"Wrote {xlsx.name} ({df['Network'].nunique()} networks, {len(df)} rows)")


def write_figures(df: pd.DataFrame) -> None:
    out = ROOT / "results_comparison"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "static_vs_iterative_auc.csv", index=False)

    # 1) Scatter: static vs iterative AUC, colored by metric, with y=x line.
    metrics = [m for m in METRIC_ORDER if m in set(df["Metric"])]
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))
    for k, m in enumerate(metrics):
        sub = df[df["Metric"] == m]
        plt.scatter(sub["Static_AUC"], sub["Iterative_AUC"],
                    label=m, color=cmap(k % 10), s=55, edgecolor="white", linewidth=0.5, alpha=0.85)
    lim = [0, max(df["Static_AUC"].max(), df["Iterative_AUC"].max()) * 1.05]
    plt.plot(lim, lim, "k--", linewidth=1, label="static = iterative")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Static AUC (lower = better)")
    plt.ylabel("Iterative AUC (lower = better)")
    plt.title("Static vs Iterative dismantling AUC\n(below diagonal = iterative more effective)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out / "static_vs_iterative_scatter.png", dpi=200)
    plt.close()

    # 2) Mean AUC per metric, static vs iterative grouped bars.
    g = df.groupby("Metric")[["Static_AUC", "Iterative_AUC"]].mean().reindex(metrics)
    x = np.arange(len(metrics)); w = 0.38
    plt.figure(figsize=(11, 6))
    plt.bar(x - w / 2, g["Static_AUC"], w, label="Static", color="#38bdf8")
    plt.bar(x + w / 2, g["Iterative_AUC"], w, label="Iterative", color="#818cf8")
    plt.xticks(x, metrics, rotation=30, ha="right")
    plt.ylabel("Mean AUC across networks (lower = better)")
    plt.title("Mean dismantling AUC by metric: static vs iterative")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out / "auc_by_metric.png", dpi=200)
    plt.close()
    print(f"Wrote results_comparison/ figures + static_vs_iterative_auc.csv")


def main() -> None:
    print("Building dashboard data from suite outputs...")
    chart_data = build()
    if not chart_data:
        print("No networks available yet — run the suite first.")
        return
    write_dashboard(chart_data)
    df = comparison_table(chart_data)
    write_xlsx(df)
    write_figures(df)
    # Console headline
    wins = df.groupby("Better").size().to_dict()
    print(f"\nStatic-vs-iterative verdict across {len(df)} (network,metric) pairs: {wins}")


if __name__ == "__main__":
    main()
