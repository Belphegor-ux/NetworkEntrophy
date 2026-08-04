"""
KFC v2 — static vs ITERATIVE dismantling on the ``All the networks/`` corpus.

The production KFC v2 here is ``rank_kfc_v2_fast`` (cluster-accelerated,
reported as ``KFC_v2``). Static ranks the real loaded graph once and removes
edges in that order; iterative recomputes the full ranking after EVERY single
removal (``get_iterative_curve``, step_size=1). All RGC curves come from
dismantling the actual graphs — no synthetic data. AUC: trapezoid over the
RGC curve, lower = better. Seeds fixed (Louvain 42).

Feasibility tiers (measured 2026-08-05, RTX 3050 / 16 GB RAM):

* ITERATIVE_NETS (16)  — static + iterative, overnight-parallel.
* STATIC_ONLY_NETS (5) — static only tonight, guarded (RAM/runtime).
* DEFERRED_NETS (1+)   — condmat: dense inverse exceeds RAM; and iterative
  on every big network. ``--estimate`` prints projected costs for later runs.

Usage (one process per network for parallelism):

    python benchmark_kfc_allnets.py --network jazz
    python benchmark_kfc_allnets.py --estimate
    python benchmark_kfc_allnets.py --aggregate

Outputs (NEW files only) under ``results_comparison/``:
    kfc_allnets_parts/<name>_{summary,curves}.csv     (per-network parts)
    kfc_v2_allnets.csv / .md                          (summary + report)
    kfc_v2_allnets_bars.png / kfc_v2_allnets_curves.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src", "utils"))
from network_utils import NetworkDismantler  # noqa: E402
from network_utils_iter import get_iterative_curve  # noqa: E402
import kfc_linalg  # noqa: E402
import prototype_criticality_v2 as pcv2  # noqa: E402

NET_DIR = os.path.join(_ROOT, "All the networks")
OUT_DIR = os.path.join(_ROOT, "results_comparison")
PARTS_DIR = os.path.join(OUT_DIR, "kfc_allnets_parts")

METRIC = "KFC_v2"                       # computed via rank_kfc_v2_fast
COLOR = "#0f8a6d"

# name -> csv filename; ordered smallest to largest (edge count of LCC).
NETWORKS: dict[str, str] = {
    "karate": "karate.csv",
    "dolphins": "dolphins.csv",
    "lesmis": "les_miserables.csv",
    "transport": "2transport_edges_re.csv",
    "football": "football.csv",
    "office": "3office_edges_re.csv",
    "baseball": "4Baseball steroid player_edges.csv",
    "crime": "5Rosenfeld crime network_edges.csv",
    "haggle": "6Haggle human proximity network_edges.csv",
    "celegans": "7celegansneural_edges.csv",
    "jazz": "JazzEdges.csv",
    "manufacturing": "8Manufacturing company email_edges.csv",
    "email": "9Email network_edges.csv",
    "wikipedia": "91Wikipedia_graph_edges.csv",
    "power": "92Power_graph_edges.csv",
    "restaurant": "93restaurant_checkin_edges.csv",
    "arxiv": "94Multilayer physicist collaborations_arXiv_edges.csv",
    "internet": "96Internet_edges.csv",
    "anybeat": "95Anybeat_graph_edges.csv",
    "astro": "97Astro_Physics_collaboration_edges.csv",
    "condmat": "98Condensed matter collaborations_edges.csv",
    "yeastnet": "99Yeastnet_v3_edgelist_numeric.csv",
}
ITERATIVE_NETS = list(NETWORKS)[:16]                # ...through restaurant
STATIC_ONLY_NETS = ["arxiv", "anybeat", "yeastnet", "astro", "internet"]
DEFERRED_NETS = ["condmat"]           # dense (N-1)^2 inverse > 16 GB RAM


def rank_v2(G: nx.Graph) -> pd.DataFrame:
    """``rank_kfc_v2_fast`` under the report name ``KFC_v2``."""
    df = pcv2.rank_kfc_v2_fast(G)
    return df.rename(columns={"KFC_v2_fast": METRIC})


def load_network(name: str) -> nx.Graph:
    path = os.path.join(NET_DIR, NETWORKS[name])
    df = pd.read_csv(path)
    if df.shape[1] < 2 or df.empty:
        raise ValueError(f"{path}: expected >=2 columns of edges, "
                         f"got shape {df.shape}")
    G = nx.from_pandas_edgelist(df, source=df.columns[0],
                                target=df.columns[1])
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        raise ValueError(f"{path}: no usable edges after cleaning")
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def _auc(rgc: list[float]) -> float:
    x = np.linspace(0, 1, len(rgc))
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(rgc, x))


def run_network(name: str) -> None:
    G = load_network(name)
    dism = NetworkDismantler(G)
    iterative = name in ITERATIVE_NETS
    gpu = "GPU" if kfc_linalg.gpu_active() else "CPU"
    print(f"=== {name}: {G.number_of_nodes()}n / {G.number_of_edges()}e "
          f"[{gpu}] ===", flush=True)

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []

    t0 = time.perf_counter()
    df = rank_v2(G)
    rgc_s = dism.get_static_curve(df, METRIC, reverse=True)
    dt_s = time.perf_counter() - t0
    modes = [("static", rgc_s, _auc(rgc_s), dt_s)]
    print(f"  static    AUC={modes[0][2]:.4f}  {dt_s:9.2f} s", flush=True)

    if iterative:
        t0 = time.perf_counter()
        rgc_i = get_iterative_curve(G, rank_v2, METRIC, reverse=True,
                                    step_size=1)
        dt_i = time.perf_counter() - t0
        modes.append(("iterative", rgc_i, _auc(rgc_i), dt_i))
        print(f"  iterative AUC={modes[1][2]:.4f}  {dt_i:9.2f} s", flush=True)

    for mode, rgc, auc, dt in modes:
        summary_rows.append({"network": name, "metric": METRIC, "mode": mode,
                             "auc": auc, "runtime_s": dt,
                             "n_points": len(rgc),
                             "n_nodes": G.number_of_nodes(),
                             "n_edges": G.number_of_edges()})
        xs = np.linspace(0, 1, len(rgc))
        curve_rows.extend({"network": name, "metric": METRIC, "mode": mode,
                           "x": float(x), "rgc": float(y)}
                          for x, y in zip(xs, rgc))

    os.makedirs(PARTS_DIR, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(PARTS_DIR, f"{name}_summary.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(
        os.path.join(PARTS_DIR, f"{name}_curves.csv"), index=False)
    print(f"Wrote parts for {name}.", flush=True)


def estimate() -> None:
    """Projected cost for deferred work: one timed static rank per network."""
    print(f"{'network':<12}{'N':>8}{'E':>9}{'static':>10}{'iter est':>12}")
    for name in STATIC_ONLY_NETS + DEFERRED_NETS:
        G = load_network(name)
        e = G.number_of_edges()
        if name in DEFERRED_NETS:
            print(f"{name:<12}{G.number_of_nodes():>8}{e:>9}"
                  f"{'>16GB RAM':>10}{'---':>12}", flush=True)
            continue
        t0 = time.perf_counter()
        rank_v2(G)
        dt = time.perf_counter() - t0
        print(f"{name:<12}{G.number_of_nodes():>8}{e:>9}{dt:>9.1f}s"
              f"{0.4 * e * dt / 3600.0:>10.1f} h", flush=True)


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _load_parts(names: list[str], kind: str) -> pd.DataFrame:
    frames = [pd.read_csv(os.path.join(PARTS_DIR, f"{n}_{kind}.csv"))
              for n in names
              if os.path.exists(os.path.join(PARTS_DIR, f"{n}_{kind}.csv"))]
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame())


def plot_bars(summary: pd.DataFrame, path: str) -> None:
    iter_nets = [n for n in ITERATIVE_NETS
                 if n in set(summary["network"])]
    static_nets = [n for n in STATIC_ONLY_NETS
                   if n in set(summary["network"])]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, 5.5),
        gridspec_kw={"width_ratios": [max(len(iter_nets), 1),
                                      max(len(static_nets), 1) * 1.6]})
    sub = summary.set_index(["network", "mode"])
    xs = np.arange(len(iter_nets))
    width = 0.38
    s_vals = [sub.loc[(n, "static"), "auc"] for n in iter_nets]
    i_vals = [sub.loc[(n, "iterative"), "auc"] for n in iter_nets]
    ax1.bar(xs - width / 2, s_vals, width, color=COLOR, alpha=0.35,
            hatch="//", edgecolor=COLOR, linewidth=1.0, label="static")
    ax1.bar(xs + width / 2, i_vals, width, color=COLOR,
            edgecolor="white", linewidth=1.0, label="iterative")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(iter_nets, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Dismantling AUC (lower = better)")
    ax1.set_title("static vs iterative (16 networks)", fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=False)

    if static_nets:
        xs2 = np.arange(len(static_nets))
        vals = [sub.loc[(n, "static"), "auc"] for n in static_nets]
        ax2.bar(xs2, vals, 0.5, color=COLOR, alpha=0.35, hatch="//",
                edgecolor=COLOR, linewidth=1.0)
        ax2.set_xticks(xs2)
        ax2.set_xticklabels(static_nets, rotation=40, ha="right", fontsize=8)
    ax2.set_title("static only (large networks)", fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("KFC_v2 (fast) — dismantling AUC on the full corpus",
                 fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close()


def plot_curves(curves: pd.DataFrame, path: str) -> None:
    nets = [n for n in ITERATIVE_NETS if n in set(curves["network"])]
    fig, axes = plt.subplots(4, 4, figsize=(18, 15), sharex=True, sharey=True)
    for ax, net in zip(axes.flat, nets):
        for mode, style, alpha in (("static", "--", 0.55),
                                   ("iterative", "-", 1.0)):
            sub = curves[(curves["network"] == net)
                         & (curves["mode"] == mode)]
            ax.plot(sub["x"], sub["rgc"], style, color=COLOR, linewidth=1.6,
                    alpha=alpha, label=mode)
        ax.set_title(net, fontweight="bold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes.flat[len(nets):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.supxlabel("Fraction of edges removed")
    fig.supylabel("Relative Giant Component")
    fig.suptitle("KFC_v2 RGC curves — dashed = static, solid = iterative",
                 fontweight="bold")
    plt.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    plt.savefig(path, dpi=150)
    plt.close()


def build_report(summary: pd.DataFrame) -> None:
    L: list[str] = []
    L.append("# KFC_v2 (fast) — static vs iterative on the full corpus\n")
    L.append("`rank_kfc_v2_fast` (cluster-accelerated KFC v2, Louvain seed "
             "42) reported as `KFC_v2`. Iterative recomputes the full edge "
             "ranking after every removal (`step_size=1`). AUC of the RGC "
             "curve, lower = better. Real dismantling of the loaded graphs "
             "throughout — no synthetic data.\n")
    L.append("## Static vs iterative (16 networks)\n")
    L.append("| network | N | E | static AUC | iterative AUC | "
             "gain (s−i) | static s | iterative s |")
    L.append("|---|---|---|---|---|---|---|---|")
    sub = summary.set_index(["network", "mode"])
    for n in ITERATIVE_NETS:
        if (n, "static") not in sub.index:
            continue
        s = sub.loc[(n, "static")]
        if (n, "iterative") in sub.index:
            i = sub.loc[(n, "iterative")]
            L.append(f"| {n} | {int(s['n_nodes'])} | {int(s['n_edges'])} | "
                     f"{s['auc']:.4f} | {i['auc']:.4f} | "
                     f"{s['auc'] - i['auc']:+.4f} | {s['runtime_s']:.2f} | "
                     f"{i['runtime_s']:.2f} |")
        else:
            L.append(f"| {n} | {int(s['n_nodes'])} | {int(s['n_edges'])} | "
                     f"{s['auc']:.4f} | — | — | {s['runtime_s']:.2f} | — |")
    got_static = [n for n in STATIC_ONLY_NETS
                  if (n, "static") in sub.index]
    L.append("\n## Static only (large networks)\n")
    L.append("| network | N | E | static AUC | runtime s |")
    L.append("|---|---|---|---|---|")
    for n in got_static:
        s = sub.loc[(n, "static")]
        L.append(f"| {n} | {int(s['n_nodes'])} | {int(s['n_edges'])} | "
                 f"{s['auc']:.4f} | {s['runtime_s']:.2f} |")
    missing = [n for n in STATIC_ONLY_NETS if n not in got_static]
    L.append("\n## Deferred\n")
    L.append("- `condmat` (36,458 nodes): the dense grounded-Laplacian "
             "inverse needs ~21 GB — exceeds this machine's 16 GB RAM. "
             "Needs an out-of-core or sparse-solver path.")
    if missing:
        L.append(f"- Static did not complete for: {', '.join(missing)} "
                 "(see run logs).")
    L.append("- Iterative on the six large networks (arxiv, internet, "
             "anybeat, astro, condmat, yeastnet): projected days-to-weeks "
             "each; run `--estimate` for current projections before "
             "attempting (GPU path via `KFC_GPU=1` applies unchanged).\n")
    L.append("## Figures\n")
    L.append("- `kfc_v2_allnets_bars.png` — AUC bars (hatched = static, "
             "solid = iterative; right panel static-only).")
    L.append("- `kfc_v2_allnets_curves.png` — 4x4 RGC grid "
             "(dashed = static, solid = iterative).")
    with open(os.path.join(OUT_DIR, "kfc_v2_allnets.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))


def aggregate() -> None:
    all_nets = ITERATIVE_NETS + STATIC_ONLY_NETS
    missing = [n for n in ITERATIVE_NETS
               if not os.path.exists(
                   os.path.join(PARTS_DIR, f"{n}_summary.csv"))]
    if missing:
        print(f"WARNING: missing iterative-tier parts: {missing}",
              flush=True)
    summary = _load_parts(all_nets, "summary")
    curves = _load_parts(all_nets, "curves")
    if summary.empty:
        raise SystemExit("No parts found — run networks first.")
    summary.to_csv(os.path.join(OUT_DIR, "kfc_v2_allnets.csv"), index=False)
    plot_bars(summary, os.path.join(OUT_DIR, "kfc_v2_allnets_bars.png"))
    plot_curves(curves, os.path.join(OUT_DIR, "kfc_v2_allnets_curves.png"))
    build_report(summary)
    print(f"Wrote kfc_v2_allnets.{{csv,md}} + bars/curves PNGs in {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", choices=list(NETWORKS))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    elif args.estimate:
        estimate()
    elif args.network:
        run_network(args.network)
    else:
        for n in ITERATIVE_NETS + STATIC_ONLY_NETS:
            try:
                run_network(n)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {n}: {exc}", flush=True)
        aggregate()


if __name__ == "__main__":
    main()
