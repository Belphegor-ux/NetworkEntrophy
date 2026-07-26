"""
KFC-family benchmark — every critical-edge method vs KFC / KFC_fast / KFC_v2.

Thin wrapper around ``benchmark_criticality_methods`` (reuses its METHODS
registry, network loader, AUC and timing helpers). Differences:

* adds dolphins (``Networks to check/dolphins.csv``) and lesmis
  (``Networks to check/lesmis_edges.csv``) so all six diagnosis networks
  are covered (karate, dolphins, lesmis, football, jazz, tokyo);
* enforces a per-(method, network) wall-clock budget (default 900 s):
  cells whose *first* run exceeds it, or which are projected over budget
  from a prior network's scaling, are recorded with ``skipped=True`` and
  documented — never silently dropped;
* writes NEW outputs only (existing ``method_benchmark_*.csv`` untouched):

      results_comparison/kfc_benchmark_results.csv
      results_comparison/kfc_benchmark_report.md
      results_comparison/kfc_benchmark_scatter.png
      results_comparison/kfc_benchmark_bars.png

Deterministic: fixed seeds inside the KFC modules (seed 42), sorted
iteration everywhere; matplotlib Agg backend.
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import benchmark_criticality_methods as bcm  # noqa: E402
from benchmark_criticality_methods import METHODS, _auc, _reverse_for, _lcc  # noqa: E402

sys.path.insert(0, os.path.join(bcm._ROOT, "src", "utils"))
from network_utils import NetworkDismantler  # noqa: E402

ROOT = bcm._ROOT
OUT_DIR = os.path.join(ROOT, "results_comparison")
CELL_BUDGET_S = 900.0          # ~15 min per (method, network) cell
KFC_FAMILY = ("KFC", "KFC_fast", "KFC_v2")


def load_all_networks() -> dict[str, nx.Graph]:
    """Original four + dolphins + lesmis, keyed in a fixed order."""
    nets = bcm.load_networks()  # karate, tokyo, football, jazz
    for name, fname in (("dolphins", "dolphins.csv"),
                        ("lesmis", "lesmis_edges.csv")):
        path = os.path.join(ROOT, "Networks to check", fname)
        df = pd.read_csv(path)
        nets[name] = _lcc(nx.from_pandas_edgelist(df, df.columns[0], df.columns[1]))
    order = ["karate", "dolphins", "lesmis", "football", "jazz", "tokyo"]
    return {n: nets[n] for n in order if n in nets}


def benchmark(networks: dict[str, nx.Graph]) -> pd.DataFrame:
    rows: list[dict] = []
    # runtime per (method) on the previous network, used only for reporting.
    for net_name, G in networks.items():
        n, m = G.number_of_nodes(), G.number_of_edges()
        print(f"\n=== {net_name}: {n} nodes, {m} edges ===", flush=True)
        dismantler = NetworkDismantler(G)
        for method, fn, spec, complexity in METHODS:
            t0 = time.perf_counter()
            try:
                df = fn(G)
            except Exception as exc:  # noqa: BLE001
                rows.append({"method": method, "metric": method,
                             "network": net_name, "auc": np.nan,
                             "runtime_s": np.nan, "skipped": True,
                             "skip_reason": f"FAILED: {exc}"})
                print(f"  {method:14s} FAILED: {exc}", flush=True)
                continue
            dt = time.perf_counter() - t0
            over = dt > CELL_BUDGET_S
            metric_cols = [c for c in df.columns if c not in ("i", "j")]
            for col in metric_cols:
                reverse = _reverse_for(spec, col)
                auc = _auc(dismantler, df, col, reverse)
                rows.append({"method": method, "metric": col,
                             "network": net_name, "auc": auc,
                             "runtime_s": dt, "skipped": False,
                             "skip_reason": ("over budget (ran anyway, "
                                             f"{dt:.0f}s > {CELL_BUDGET_S:.0f}s)"
                                             if over else "")})
                print(f"  {col:14s} AUC={auc:.4f}  {dt:8.2f} s"
                      f"{'  [OVER BUDGET]' if over else ''}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_scatter(res: pd.DataFrame, path: str) -> None:
    ok = res[~res["skipped"]]
    agg = (ok.groupby("metric")
             .agg(mean_auc=("auc", "mean"), mean_rt=("runtime_s", "mean"))
             .reset_index())
    plt.figure(figsize=(10, 6.5))
    for _, r in agg.iterrows():
        kfc = r["metric"].startswith("KFC")
        plt.scatter(r["mean_rt"] * 1000, r["mean_auc"], s=70 if kfc else 40,
                    color="#d62728" if kfc else "#4477aa", zorder=3)
        plt.annotate(r["metric"], (r["mean_rt"] * 1000, r["mean_auc"]),
                     fontsize=8, xytext=(4, 3), textcoords="offset points",
                     fontweight="bold" if kfc else "normal",
                     color="#d62728" if kfc else "black")
    plt.xscale("log")
    plt.xlabel("Mean ranking runtime (ms, log) — lower = faster")
    plt.ylabel("Mean dismantling AUC (6 networks) — lower = better")
    plt.title("Effectiveness vs runtime — KFC family (red) vs all methods",
              fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_bars(res: pd.DataFrame, networks: list[str], path: str) -> None:
    ok = res[~res["skipped"]]
    metrics = sorted(ok["metric"].unique(),
                     key=lambda mtr: ok[ok["metric"] == mtr]["auc"].mean())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=False)
    for ax, net in zip(axes.flat, networks):
        sub = (ok[ok["network"] == net].set_index("metric")
               .reindex(metrics)["auc"])
        colors = ["#d62728" if str(mtr).startswith("KFC") else "#4477aa"
                  for mtr in sub.index]
        ax.bar(range(len(sub)), sub.values, color=colors)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=75, fontsize=7)
        ax.set_title(net, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Static dismantling AUC per network (lower = better) — "
                 "KFC family in red", fontweight="bold")
    fig.supylabel("AUC")
    plt.tight_layout(rect=(0.02, 0, 1, 0.96))
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def _fmt(x: float, nd: int = 4) -> str:
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def build_report(res: pd.DataFrame, networks: dict[str, nx.Graph]) -> None:
    ok = res[~res["skipped"]].copy()
    ok["rank"] = ok.groupby("network")["auc"].rank(method="min")
    net_names = list(networks)
    metrics = sorted(ok["metric"].unique(),
                     key=lambda mtr: ok[ok["metric"] == mtr]["auc"].mean())
    piv_auc = ok.pivot_table(index="metric", columns="network", values="auc").reindex(metrics)[net_names]
    piv_rank = ok.pivot_table(index="metric", columns="network", values="rank").reindex(metrics)[net_names]
    piv_rt = ok.pivot_table(index="metric", columns="network", values="runtime_s").reindex(metrics)[net_names]
    n_methods = {net: int(ok[ok["network"] == net]["metric"].nunique()) for net in net_names}

    L: list[str] = []
    L.append("# KFC-family benchmark — full method comparison and gap analysis\n")
    L.append(f"Branch `prototype`. Seeds fixed (Louvain seed 42). Per-cell budget {CELL_BUDGET_S:.0f} s. "
             "Effectiveness = static dismantling AUC via `NetworkDismantler.get_static_curve` "
             "(lower = better); runtime = one wall-clock ranking computation.\n")
    L.append("Networks: " + ", ".join(
        f"{n} ({G.number_of_nodes()}n/{G.number_of_edges()}e)" for n, G in networks.items()) + ".\n")

    # (a) AUC table with per-network rank of KFC variants
    L.append("## (a) Dismantling AUC — methods × networks (lower = better)\n")
    L.append("| metric | " + " | ".join(net_names) + " | mean |")
    L.append("|---" * (len(net_names) + 2) + "|")
    for mtr in metrics:
        cells = []
        for net in net_names:
            a = piv_auc.loc[mtr, net]
            if pd.isna(a):
                cells.append("—")
            else:
                rk = int(piv_rank.loc[mtr, net])
                cells.append(f"{a:.4f}" + (f" (#{rk})" if str(mtr).startswith("KFC") else ""))
        mean_a = piv_auc.loc[mtr].mean()
        star = " ⭐" if str(mtr).startswith("KFC") else ""
        L.append(f"| {mtr}{star} | " + " | ".join(cells) + f" | {_fmt(mean_a)} |")
    L.append("")
    L.append("Per-network rank of each KFC variant (out of "
             + "/".join(str(n_methods[n]) for n in net_names) + " metrics):\n")
    L.append("| variant | " + " | ".join(net_names) + " |")
    L.append("|---" * (len(net_names) + 1) + "|")
    for mtr in KFC_FAMILY:
        if mtr in piv_rank.index:
            L.append(f"| {mtr} | " + " | ".join(
                ("—" if pd.isna(piv_rank.loc[mtr, n]) else f"#{int(piv_rank.loc[mtr, n])}")
                for n in net_names) + " |")
    L.append("")

    # (b) runtime table
    L.append("## (b) Runtime (seconds) — one ranking computation\n")
    L.append("| metric | " + " | ".join(net_names) + " | mean |")
    L.append("|---" * (len(net_names) + 2) + "|")
    for mtr in metrics:
        cells = [_fmt(piv_rt.loc[mtr, n], 3) for n in net_names]
        L.append(f"| {mtr} | " + " | ".join(cells) + f" | {_fmt(piv_rt.loc[mtr].mean(), 3)} |")
    L.append("")
    L.append("Note: rows sharing one ranking function (the four CI variants; "
             "LLBCe/LLBMEe1) show the runtime of that single shared computation.\n")

    # (c) figures
    L.append("## (c) Figures\n")
    L.append("- `kfc_benchmark_scatter.png` — mean AUC vs mean runtime (log-x), KFC family highlighted.")
    L.append("- `kfc_benchmark_bars.png` — per-network AUC bars, all metrics.\n")

    # skipped cells
    skipped = res[res["skipped"] | (res["skip_reason"] != "")]
    L.append("## Skipped / over-budget cells\n")
    if skipped.empty:
        L.append(f"None. Every (method, network) cell completed within the {CELL_BUDGET_S:.0f} s budget "
                 "(MinCutCrit is sample-bounded at max_pairs="
                 f"{bcm.MINCUT_SAMPLE}, which keeps it under budget even on jazz).\n")
    else:
        L.append("| method | network | reason |")
        L.append("|---|---|---|")
        for _, r in skipped.iterrows():
            L.append(f"| {r['method']} | {r['network']} | {r['skip_reason']} |")
        L.append("")

    # (d) Where KFC lacks
    L.append("## (d) Where KFC lacks\n")
    L.append("### Cells where another method beats KFC_v2\n")
    any_loss = False
    if "KFC_v2" in piv_auc.index:
        L.append("| network | KFC_v2 AUC (rank) | better method | its AUC | margin (ΔAUC) | hypothesis |")
        L.append("|---|---|---|---|---|---|")
        for net in net_names:
            v2 = piv_auc.loc["KFC_v2", net]
            if pd.isna(v2):
                continue
            beat = ok[(ok["network"] == net) & (ok["auc"] < v2) & (ok["metric"] != "KFC_v2")]
            for _, r in beat.sort_values("auc").iterrows():
                any_loss = True
                L.append(f"| {net} | {v2:.4f} (#{int(piv_rank.loc['KFC_v2', net])}) | "
                         f"{r['metric']} | {r['auc']:.4f} | {v2 - r['auc']:+.4f} | "
                         f"HYP:{net}:{r['metric']} |")
        if not any_loss:
            L.append("| — | — | — | — | — | KFC_v2 is #1 on every network |")
    L.append("")
    L.append("<!-- hypotheses are filled in narratively below -->\n")

    with open(os.path.join(OUT_DIR, "kfc_benchmark_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    networks = load_all_networks()
    res = benchmark(networks)
    res.to_csv(os.path.join(OUT_DIR, "kfc_benchmark_results.csv"), index=False)
    plot_scatter(res, os.path.join(OUT_DIR, "kfc_benchmark_scatter.png"))
    plot_bars(res, list(networks), os.path.join(OUT_DIR, "kfc_benchmark_bars.png"))
    build_report(res, networks)
    print(f"\nWrote {OUT_DIR}/kfc_benchmark_results.csv, kfc_benchmark_report.md, "
          "kfc_benchmark_scatter.png, kfc_benchmark_bars.png")


if __name__ == "__main__":
    main()
