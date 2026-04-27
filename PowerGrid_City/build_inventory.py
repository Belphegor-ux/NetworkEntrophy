"""Generate a detailed markdown inventory of the Tokyo Power Grid.

Produces ``results_city/tokyo_grid_inventory.md``: per-node table with name,
coordinates, degree, k-shell, neighbors, and per-edge LLBCe ranks.

Read-only: consumes ``datasets/tokyo_grid.gml``, ``datasets/japan_nodes.csv``,
and ``results_city/tokyo_grid_metrics.csv``. Writes only the markdown file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GML_PATH = os.path.join(PROJECT_ROOT, "PowerGrid_City", "datasets", "tokyo_grid.gml")
NODES_CSV = os.path.join(PROJECT_ROOT, "PowerGrid_City", "datasets", "japan_nodes.csv")
METRICS_CSV = os.path.join(PROJECT_ROOT, "results_city", "tokyo_grid_metrics.csv")
OUT_MD = os.path.join(PROJECT_ROOT, "results_city", "tokyo_grid_inventory.md")


@dataclass(frozen=True)
class NodeRow:
    node_id: int
    v_id: int
    name: str
    lon: float
    lat: float
    voltage: str
    operator: str
    degree: int
    kshell: int


def _classify_region(lon: float, lat: float) -> str:
    """Coarse Kanto sub-region classification by lon/lat box.

    Buckets are conservative — a point at the boundary stays in the more
    populated bucket. Only used for the regional summary table; the
    per-node table reports raw lon/lat.
    """
    if lat >= 36.3:
        return "North Kanto (Gunma/Tochigi/Ibaraki N)"
    if lat >= 35.9 and lon <= 139.6:
        return "NW Kanto (Saitama/W Tokyo)"
    if lat >= 35.9 and lon > 139.6:
        return "NE Kanto (Ibaraki S/N Chiba)"
    if 35.5 <= lat < 35.9 and lon <= 139.6:
        return "Tokyo Metro (W)"
    if 35.5 <= lat < 35.9 and lon > 139.6:
        return "Tokyo Metro (E) / Chiba"
    if lat < 35.5 and lon <= 139.6:
        return "South Kanto (Kanagawa/W Izu)"
    return "South Kanto (S Chiba/Boso)"


def load_graph() -> nx.Graph:
    g = nx.read_gml(GML_PATH)
    return nx.convert_node_labels_to_integers(g, label_attribute="label")


def load_nodes_df() -> pd.DataFrame:
    df = pd.read_csv(NODES_CSV, sep="#")
    return df[["v_id", "lon", "lat", "name", "voltage", "operator", "typ"]].copy()


def load_metrics_df() -> pd.DataFrame:
    return pd.read_csv(METRICS_CSV)


def _v_id(graph: nx.Graph, node: int) -> Optional[int]:
    raw = graph.nodes[node].get("label")
    if raw is None:
        return None
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def build_node_rows(
    graph: nx.Graph, nodes_df: pd.DataFrame
) -> Tuple[List[NodeRow], int]:
    lookup = {
        int(r.v_id): (
            float(r.lon),
            float(r.lat),
            "" if pd.isna(r.name) else str(r.name).strip(),
            "" if pd.isna(r.voltage) else str(r.voltage).strip(),
            "" if pd.isna(r.operator) else str(r.operator).strip(),
            "" if pd.isna(r.typ) else str(r.typ).strip(),
        )
        for r in nodes_df.itertuples(index=False)
    }
    core = nx.core_number(graph)
    rows: List[NodeRow] = []
    skipped = 0
    for n in graph.nodes():
        v_id = _v_id(graph, n)
        if v_id is None or v_id not in lookup:
            skipped += 1
            continue
        lon, lat, name, volt, op, typ = lookup[v_id]
        display_name = name if name else f"(unnamed {typ or 'node'})"
        rows.append(
            NodeRow(
                node_id=int(n),
                v_id=v_id,
                name=display_name,
                lon=lon,
                lat=lat,
                voltage=volt,
                operator=op,
                degree=int(graph.degree(n)),
                kshell=int(core.get(n, 0)),
            )
        )
    rows.sort(key=lambda r: (-r.degree, r.name))
    return rows, skipped


def edge_llbce_rank(metrics: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[float, int]]:
    """Map sorted (i, j) -> (LLBCe value, 1-based rank where 1 == most critical)."""
    df = metrics[["i", "j", "LLBCe"]].copy()
    df["key"] = df.apply(lambda r: tuple(sorted((int(r.i), int(r.j)))), axis=1)
    df = df.sort_values("LLBCe", ascending=False).reset_index(drop=True)
    out: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        out[row.key] = (float(row.LLBCe), rank)
    return out


def render_markdown(
    graph: nx.Graph,
    rows: List[NodeRow],
    metrics: pd.DataFrame,
    skipped: int,
) -> str:
    name_by_node: Dict[int, str] = {r.node_id: r.name for r in rows}
    edge_lookup = edge_llbce_rank(metrics)

    lines: List[str] = []
    lines.append("# Tokyo Power Grid — Node and Edge Inventory")
    lines.append("")
    lines.append(
        "Auto-generated from `datasets/tokyo_grid.gml`, "
        "`datasets/japan_nodes.csv`, and `results_city/tokyo_grid_metrics.csv`. "
        "Re-run via `python PowerGrid_City/build_inventory.py`."
    )
    lines.append("")

    # Header summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Nodes (LCC)**: {graph.number_of_nodes()}")
    lines.append(f"- **Edges (LCC)**: {graph.number_of_edges()}")
    lines.append(f"- **Named (substations or labelled junctions)**: {sum(1 for r in rows if not r.name.startswith('(unnamed'))}")
    lines.append(f"- **Unnamed (OSM line-segment vertices)**: {sum(1 for r in rows if r.name.startswith('(unnamed'))}")
    if skipped:
        lines.append(f"- **Skipped (no coordinate match in japan_nodes.csv)**: {skipped}")
    lines.append("")

    # Regional breakdown
    region_counts: Dict[str, Dict[str, int]] = {}
    region_edges: Dict[str, int] = {}
    region_to_nodes: Dict[str, set] = {}
    for r in rows:
        region = _classify_region(r.lon, r.lat)
        region_counts.setdefault(region, {"nodes": 0, "named": 0})
        region_counts[region]["nodes"] += 1
        if not r.name.startswith("(unnamed"):
            region_counts[region]["named"] += 1
        region_to_nodes.setdefault(region, set()).add(r.node_id)
    for u, v in graph.edges():
        rn_u = next(
            (region for region, ids in region_to_nodes.items() if u in ids),
            None,
        )
        rn_v = next(
            (region for region, ids in region_to_nodes.items() if v in ids),
            None,
        )
        if rn_u and rn_v and rn_u == rn_v:
            region_edges[rn_u] = region_edges.get(rn_u, 0) + 1
        elif rn_u and rn_v:
            key = "Inter-region"
            region_edges[key] = region_edges.get(key, 0) + 1

    lines.append("## Regional Breakdown")
    lines.append("")
    lines.append("| Region | Nodes | Named substations | Intra-region edges |")
    lines.append("|---|---:|---:|---:|")
    for region in sorted(region_counts.keys()):
        c = region_counts[region]
        lines.append(
            f"| {region} | {c['nodes']} | {c['named']} | "
            f"{region_edges.get(region, 0)} |"
        )
    if "Inter-region" in region_edges:
        lines.append(
            f"| _Inter-region edges_ | — | — | "
            f"{region_edges['Inter-region']} |"
        )
    lines.append("")

    # Top hubs (by degree)
    lines.append("## Top 15 Hubs (by degree)")
    lines.append("")
    lines.append("| Rank | Name | Voltage (V) | Degree | k-shell | Lon | Lat | Operator |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows[:15], start=1):
        v_str = r.voltage.split(";")[0] if r.voltage else "—"
        op_str = r.operator if r.operator else "—"
        lines.append(
            f"| {i} | {r.name} | {v_str} | {r.degree} | {r.kshell} | "
            f"{r.lon:.4f} | {r.lat:.4f} | {op_str} |"
        )
    lines.append("")

    # Full inventory
    lines.append("## Full Node Inventory (sorted by degree desc, then name)")
    lines.append("")
    lines.append("| Node ID | v_id | Name | Lon | Lat | Voltage (V) | Operator | Degree | k-shell | Neighbors (node_id : name) | Incident edges (LLBCe rank) |")
    lines.append("|---:|---:|---|---:|---:|---|---|---:|---:|---|---|")
    for r in rows:
        neighbors = sorted(graph.neighbors(r.node_id))
        nb_str = "; ".join(
            f"{nb}: {name_by_node.get(nb, '?')}" for nb in neighbors
        )
        edge_str_parts: List[str] = []
        for nb in neighbors:
            key = tuple(sorted((r.node_id, nb)))
            llbce_val, llbce_rank = edge_lookup.get(key, (float("nan"), -1))
            if llbce_rank > 0:
                edge_str_parts.append(
                    f"({r.node_id},{nb})#{llbce_rank} ({llbce_val:.3g})"
                )
        edge_str = "; ".join(edge_str_parts) if edge_str_parts else "—"
        v_str = r.voltage.split(";")[0] if r.voltage else "—"
        op_str = r.operator if r.operator else "—"
        # Escape pipes in name
        name_safe = r.name.replace("|", "\\|")
        lines.append(
            f"| {r.node_id} | {r.v_id} | {name_safe} | {r.lon:.4f} | "
            f"{r.lat:.4f} | {v_str} | {op_str} | {r.degree} | {r.kshell} | "
            f"{nb_str} | {edge_str} |"
        )

    lines.append("")
    lines.append("## Edge Inventory (top 30 most critical by LLBCe)")
    lines.append("")
    lines.append("| Rank | i | j | i name | j name | LLBCe | LDC | Jaccard | LKS |")
    lines.append("|---:|---:|---:|---|---|---:|---:|---:|---:|")
    metrics_top = (
        metrics[["i", "j", "LDC", "Jaccard", "LKS", "LLBCe"]]
        .sort_values("LLBCe", ascending=False)
        .head(30)
    )
    for rank, row in enumerate(metrics_top.itertuples(index=False), start=1):
        i_name = name_by_node.get(int(row.i), "?")
        j_name = name_by_node.get(int(row.j), "?")
        lines.append(
            f"| {rank} | {int(row.i)} | {int(row.j)} | {i_name} | {j_name} | "
            f"{row.LLBCe:.4g} | {row.LDC:.0f} | {row.Jaccard:.3g} | "
            f"{row.LKS:.0f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    print(f"[build_inventory] Loading graph: {GML_PATH}")
    graph = load_graph()
    print(f"[build_inventory] Graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")

    print(f"[build_inventory] Loading nodes: {NODES_CSV}")
    nodes_df = load_nodes_df()

    print(f"[build_inventory] Loading metrics: {METRICS_CSV}")
    metrics_df = load_metrics_df()

    rows, skipped = build_node_rows(graph, nodes_df)
    print(f"[build_inventory] Resolved {len(rows)} nodes "
          f"(skipped {skipped})")

    md = render_markdown(graph, rows, metrics_df, skipped)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[build_inventory] Wrote: {OUT_MD}  ({len(md):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
