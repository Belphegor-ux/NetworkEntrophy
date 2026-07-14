"""
Tokyo Power Grid — resilience / criticality analysis (max-flow, min-cut,
current-flow, spectral).

Purpose: rank the edges and nodes of the Tokyo high-voltage backbone by how
critical they are to keeping the network connected, so they can be
**prioritised for protection, redundancy and N-1 contingency planning**. This
is the flow/spectral complement to the topological dismantling metrics produced
by ``run_city_analysis.py``.

Outputs (written to results_city/, never touching tokyo_grid_metrics.csv):
    tokyo_resilience_metrics.csv   per-edge: EBC, CFEdge, MinCutCrit, BridgeImpact
                                   (+ joined line name / voltage / length)
    tokyo_node_resilience.csv      per-node: degree, betweenness, current-flow,
                                   articulation-point / min-node-cut flags
                                   (+ joined substation name / voltage / freq)
    tokyo_resilience_report.md     global min-cut / connectivity / spectral
                                   summary + top-K protection priorities
    plot_resilience_all.png        dismantling (RGC vs fraction removed) for
                                   every resilience metric, with AUCs

Substation / line names are joined read-only from datasets/japan_nodes.csv and
datasets/japan_edges.csv (per the repo constraint: read data, never modify it).
"""
import os
import sys

import networkx as nx
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "utils"))

from network_utils import run_and_plot  # noqa: E402
import resilience_utils as ru  # noqa: E402

GML_PATH = os.path.join(_HERE, "datasets", "tokyo_grid.gml")
NODES_CSV = os.path.join(_HERE, "datasets", "japan_nodes.csv")
EDGES_CSV = os.path.join(_HERE, "datasets", "japan_edges.csv")
RESULTS_DIR = os.path.join(_ROOT, "results_city")


def _load_graph() -> nx.Graph:
    """Load the Tokyo grid keeping original v_id labels (for name joins)."""
    if not os.path.exists(GML_PATH):
        raise FileNotFoundError(
            f"{GML_PATH} not found. Run download_and_filter_tokyo.py first."
        )
    G = nx.read_gml(GML_PATH)
    return ru.largest_connected_component(G)


def _load_node_names() -> dict[str, dict]:
    """Map v_id -> {name, voltage, frequency, typ} from japan_nodes.csv (read-only)."""
    try:
        df = pd.read_csv(NODES_CSV, sep="#", dtype=str, keep_default_na=False,
                         engine="python")
    except Exception as exc:  # noqa: BLE001
        print(f"  [names] could not read {NODES_CSV}: {exc}")
        return {}
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        out[str(r.get("v_id", "")).strip()] = {
            "name": r.get("name", "").strip(),
            "voltage": r.get("voltage", "").strip(),
            "frequency": r.get("frequency", "").strip(),
            "typ": r.get("typ", "").strip(),
        }
    return out


def _load_edge_names() -> dict[tuple, dict]:
    """Map canonical (v_id_1, v_id_2) -> line metadata from japan_edges.csv."""
    try:
        df = pd.read_csv(EDGES_CSV, sep="#", dtype=str, keep_default_na=False,
                         engine="python")
    except Exception as exc:  # noqa: BLE001
        print(f"  [names] could not read {EDGES_CSV}: {exc}")
        return {}
    out: dict[tuple, dict] = {}
    for _, r in df.iterrows():
        key = ru._canon(str(r.get("v_id_1", "")).strip(),
                        str(r.get("v_id_2", "")).strip())
        out[key] = {
            "line_name": r.get("name", "").strip(),
            "line_voltage": r.get("voltage", "").strip(),
            "length_m": r.get("length_m", "").strip(),
        }
    return out


def _fmt_name(meta: dict | None, fallback: str) -> str:
    if not meta:
        return fallback
    name = meta.get("name") or meta.get("line_name") or ""
    volt = meta.get("voltage") or meta.get("line_voltage") or ""
    volt_kv = f"{int(volt) // 1000}kV" if volt.isdigit() else ""
    label = name if name else fallback
    return f"{label} ({volt_kv})" if volt_kv else label


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    G = _load_graph()
    print(f"Tokyo grid (LCC): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_names = _load_node_names()
    edge_names = _load_edge_names()

    # ---- Edge-level criticality --------------------------------------------
    print("Computing edge criticality (EBC, current-flow, min-cut, N-1 impact)...")
    edge_df = ru.combined_edge_criticality(G)
    # Join line metadata.
    meta_rows = []
    for _, row in edge_df.iterrows():
        m = edge_names.get(ru._canon(row["i"], row["j"]), {})
        meta_rows.append(m)
    meta_df = pd.DataFrame(meta_rows, index=edge_df.index)
    edge_out = pd.concat([edge_df, meta_df], axis=1)
    edge_csv = os.path.join(RESULTS_DIR, "tokyo_resilience_metrics.csv")
    edge_out.to_csv(edge_csv, index=False)
    print(f"  -> {edge_csv}")

    # ---- Node-level criticality --------------------------------------------
    print("Computing node criticality...")
    node_df = ru.node_resilience_table(G)
    node_df["name"] = node_df["node"].map(lambda v: node_names.get(str(v), {}).get("name", ""))
    node_df["voltage"] = node_df["node"].map(lambda v: node_names.get(str(v), {}).get("voltage", ""))
    node_df["frequency"] = node_df["node"].map(lambda v: node_names.get(str(v), {}).get("frequency", ""))
    node_csv = os.path.join(RESULTS_DIR, "tokyo_node_resilience.csv")
    node_df.to_csv(node_csv, index=False)
    print(f"  -> {node_csv}")

    # ---- Dismantling curves (AUC comparison) -------------------------------
    # Reuse the already-computed metrics (the all-pairs min-cut is the most
    # expensive step; recomputing it inside run_and_plot would double the run).
    # Passing the clean 4-metric frame (no joined metadata columns) also keeps
    # run_and_plot's "every non-i/j column is a metric" inference correct.
    print("Running resilience dismantling curves...")
    auc_results = run_and_plot(
        G, "Tokyo Resilience Metrics", lambda _g: edge_df,
        os.path.join(RESULTS_DIR, "plot_resilience_all.png"), reverse=True,
    )

    # ---- Global summary + report -------------------------------------------
    print("Computing global min-cut / connectivity / spectral summary...")
    summary = ru.global_resilience_summary(G)
    report_path = os.path.join(RESULTS_DIR, "tokyo_resilience_report.md")
    _write_report(report_path, summary, edge_out, node_df, auc_results,
                  edge_names, node_names)
    print(f"  -> {report_path}")
    print("Resilience analysis complete.")


def _top_edges(edge_out: pd.DataFrame, col: str, edge_names: dict, k: int = 10) -> str:
    top = edge_out.sort_values(col, ascending=False).head(k)
    lines = [f"| rank | edge (v_id) | line | {col} |", "|---|---|---|---|"]
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        label = _fmt_name(edge_names.get(ru._canon(r["i"], r["j"])),
                          f"{r['i']}–{r['j']}")
        lines.append(f"| {rank} | {r['i']}–{r['j']} | {label} | {r[col]:.4f} |")
    return "\n".join(lines)


def _top_nodes(node_df: pd.DataFrame, col: str, k: int = 10) -> str:
    top = node_df.sort_values(col, ascending=False).head(k)
    lines = [f"| rank | node (v_id) | name | {col} | AP? |", "|---|---|---|---|---|"]
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        nm = r.get("name", "") or "—"
        ap = "yes" if r["is_articulation_point"] else ""
        lines.append(f"| {rank} | {r['node']} | {nm} | {r[col]:.4f} | {ap} |")
    return "\n".join(lines)


def _write_report(path, summary, edge_out, node_df, auc_results,
                  edge_names, node_names) -> None:
    L: list[str] = []
    L.append("# Tokyo Power Grid — Resilience & Critical-Component Report\n")
    L.append(
        "**Purpose.** This report ranks the edges (transmission lines) and nodes "
        "(substations / junctions) of the Tokyo high-voltage backbone by how "
        "critical they are to global connectivity, so operators can prioritise "
        "them for **protection, redundancy, and N-1 / N-k contingency planning**. "
        "It is the flow/spectral complement to the topological dismantling suite "
        "in `tokyo_grid_metrics.csv`.\n"
    )
    L.append(
        "> **Data caveat.** The graph is an OpenStreetMap-derived extract "
        "(156 nodes, 190 edges, largest connected component). Many nodes are "
        "line-junction vertices rather than named substations, and the topology "
        "carries no per-line electrical parameters, so current-flow metrics use "
        "unit conductances (a topological electrical model, not a full AC "
        "power-flow). Treat rankings as structural indicators, not operational "
        "targeting.\n"
    )

    L.append("## 1. Global robustness summary\n")
    L.append(f"- **Edge connectivity λ(G)** = {summary['edge_connectivity']} "
             "(minimum number of lines whose loss disconnects the network)")
    L.append(f"- **Node connectivity κ(G)** = {summary['node_connectivity']}")
    L.append(f"- **Global min-cut (Stoer–Wagner)** = {summary['global_min_cut_value']}, "
             f"partition sizes {summary['global_min_cut_partition_sizes']}")
    bmc = summary.get("backbone_min_cut_value")
    if bmc is not None:
        L.append(f"- **Backbone (2-core) min-cut** = {bmc}, "
                 f"partition sizes {summary.get('backbone_min_cut_partition_sizes')} "
                 f"(over {summary.get('backbone_n_nodes')} backbone nodes) — the "
                 "meaningful structural bottleneck once dead-end stub lines are pruned")
    L.append(f"- **Algebraic connectivity (Fiedler value)** = "
             f"{summary['algebraic_connectivity']:.5f} "
             "(global robustness index; near 0 = close to fragmenting)")
    L.append(f"- **Fiedler bisection** splits the grid into "
             f"{summary['fiedler_partition_sizes']} nodes")
    L.append(f"- **Bridges** (single-line cut edges): {summary['n_bridges']}")
    L.append(f"- **Articulation points** (single-node cut vertices): "
             f"{summary['n_articulation_points']}")
    if summary.get("diameter") is not None:
        L.append(f"- **Diameter**: {summary['diameter']} hops")
    L.append(
        "\nInterpretation: λ = κ = 1 and a raw global min-cut of 1 reflect the "
        "many pendant stub lines in the extract (the cheapest cut merely isolates "
        "one leaf). The **backbone min-cut** and the **bridge/N-1 rankings below** "
        "are the actionable resilience signal.\n"
    )

    L.append("## 2. Dismantling effectiveness (AUC — lower = ranks critical edges better)\n")
    L.append("| metric | dismantling AUC |")
    L.append("|---|---|")
    for col, res in sorted(auc_results.items(), key=lambda kv: kv[1]["auc"]):
        L.append(f"| {col} | {res['auc']:.4f} |")
    L.append(
        "\n_Note: `BridgeImpact` is a single-failure (N-1) indicator — it is 0 for "
        "every non-bridge, so it is deliberately not a full dismantling order and "
        "its higher AUC here is expected. Use it as a protection flag (Section 3), "
        "not as a sequential ranking._\n"
    )

    L.append("## 3. Top critical lines to protect (N-1 single-failure impact)\n")
    L.append("Edges ranked by the fraction of the network severed if that one "
             "line is lost (bridges only; non-bridges = 0).\n")
    L.append(_top_edges(edge_out, "BridgeImpact", edge_names))
    L.append("")

    L.append("## 4. Top bottleneck lines (min-cut membership via max-flow)\n")
    L.append("Edges lying on the minimum source–sink cut for the largest share "
             "of node pairs.\n")
    L.append(_top_edges(edge_out, "MinCutCrit", edge_names))
    L.append("")

    L.append("## 5. Top load-bearing corridors (current-flow betweenness)\n")
    L.append("Edges carrying the most electrical throughput under a unit-current "
             "resistor model.\n")
    L.append(_top_edges(edge_out, "CFEdge", edge_names))
    L.append("")

    L.append("## 6. Top critical substations / junctions to protect\n")
    L.append("Nodes ranked by current-flow betweenness; articulation points are "
             "single-node cut vertices whose loss disconnects the network.\n")
    L.append(_top_nodes(node_df, "CFNode"))
    L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
