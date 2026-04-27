"""Render the Tokyo Power Grid onto a real geographic Kanto map.

Produces two outputs in ``results_city/``:

* ``kanto_map.html`` — interactive folium map (OpenStreetMap tiles).
* ``kanto_map.png`` — static matplotlib image with a contextily basemap.

Geographic coordinates are sourced from ``datasets/japan_nodes.csv``
(no synthesized values). The graph is read from ``datasets/tokyo_grid.gml``
and edge ranks come from ``results_city/tokyo_grid_metrics.csv``.

Read-only with respect to all data files.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from os.path import abspath, dirname, exists, join
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: str = dirname(dirname(abspath(__file__)))
GML_PATH: str = join(PROJECT_ROOT, "PowerGrid_City", "datasets", "tokyo_grid.gml")
NODES_CSV: str = join(PROJECT_ROOT, "PowerGrid_City", "datasets", "japan_nodes.csv")
METRICS_CSV: str = join(PROJECT_ROOT, "results_city", "tokyo_grid_metrics.csv")
OUT_HTML: str = join(PROJECT_ROOT, "results_city", "kanto_map.html")
OUT_PNG: str = join(PROJECT_ROOT, "results_city", "kanto_map.png")

KANTO_CENTER: Tuple[float, float] = (35.7, 139.7)
TITLE: str = "Tokyo Power Grid - Critical Link Analysis (LLBCe ranked)"

TRUNK_HUBS: Tuple[str, ...] = (
    "Shin-Tokorozawa",
    "Shin-Tama",
    "Shin-Keiyo",
    "Shin-Hadano",
    "Shin-Okabe",
    "Shin-Haruna",
    "Shin-Fuji",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeView:
    """Immutable rendering record for a single node."""

    node_id: int
    v_id: int
    lon: float
    lat: float
    name: str
    degree: int
    kshell: int


@dataclass(frozen=True)
class EdgeView:
    """Immutable rendering record for a single edge."""

    u: int
    v: int
    lon_u: float
    lat_u: float
    lon_v: float
    lat_v: float
    llbce: float
    llbce_rank_pct: float  # 0.0 (lowest) … 1.0 (highest)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_graph(gml_path: str) -> nx.Graph:
    """Load the Tokyo grid GML and convert node labels to integers.

    Mirrors the convention used in :mod:`run_city_analysis`: after
    ``convert_node_labels_to_integers``, each node carries its original
    OSM ``v_id`` in the ``label`` attribute as a string.
    """
    if not exists(gml_path):
        raise FileNotFoundError(f"GML not found: {gml_path}")
    graph = nx.read_gml(gml_path)
    return nx.convert_node_labels_to_integers(graph, label_attribute="label")


def load_node_coords(csv_path: str) -> pd.DataFrame:
    """Load OSM node CSV with columns ``v_id, lon, lat, ..., name, ...``.

    The CSV is ``#``-separated (per upstream ComplexNetTSP convention).
    """
    if not exists(csv_path):
        raise FileNotFoundError(f"Nodes CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, sep="#")
    needed = {"v_id", "lon", "lat", "name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Nodes CSV missing required columns: {missing}")
    return df[["v_id", "lon", "lat", "name"]].copy()


def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load edge metrics; require ``i, j, LLBCe``."""
    if not exists(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"i", "j", "LLBCe"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV missing required columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Building views
# ---------------------------------------------------------------------------


def _node_v_id(graph: nx.Graph, node: int) -> Optional[int]:
    """Return the original OSM ``v_id`` for *node*, or ``None`` if absent."""
    raw = graph.nodes[node].get("label")
    if raw is None:
        return None
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def build_node_views(
    graph: nx.Graph, nodes_df: pd.DataFrame
) -> Tuple[Tuple[NodeView, ...], int]:
    """Build node views, returning views plus the count of nodes skipped."""
    coord_lookup: Dict[int, Tuple[float, float, str]] = {
        int(row.v_id): (
            float(row.lon),
            float(row.lat),
            "" if pd.isna(row.name) else str(row.name),
        )
        for row in nodes_df.itertuples(index=False)
    }

    core: Dict[int, int] = nx.core_number(graph)
    views: List[NodeView] = []
    skipped: int = 0

    for node in graph.nodes():
        v_id = _node_v_id(graph, node)
        if v_id is None or v_id not in coord_lookup:
            skipped += 1
            continue
        lon, lat, name = coord_lookup[v_id]
        views.append(
            NodeView(
                node_id=int(node),
                v_id=v_id,
                lon=lon,
                lat=lat,
                name=name,
                degree=int(graph.degree(node)),
                kshell=int(core.get(node, 0)),
            )
        )

    if skipped:
        print(
            f"[render_kanto_map] WARNING: skipped {skipped} node(s) without "
            "coordinates in japan_nodes.csv",
            file=sys.stderr,
        )
    return tuple(views), skipped


def build_edge_views(
    graph: nx.Graph,
    metrics: pd.DataFrame,
    node_views: Iterable[NodeView],
) -> Tuple[EdgeView, ...]:
    """Compose per-edge views with LLBCe rank percentile in [0, 1]."""
    coords: Dict[int, Tuple[float, float]] = {
        nv.node_id: (nv.lon, nv.lat) for nv in node_views
    }

    metric_lookup: Dict[Tuple[int, int], float] = {}
    for row in metrics.itertuples(index=False):
        key = tuple(sorted((int(row.i), int(row.j))))
        metric_lookup[key] = float(row.LLBCe)

    pairs: List[Tuple[int, int, float]] = []
    for u, v in graph.edges():
        if u not in coords or v not in coords:
            continue
        key = tuple(sorted((int(u), int(v))))
        score = metric_lookup.get(key)
        if score is None:
            score = float("nan")
        pairs.append((int(u), int(v), score))

    if not pairs:
        return tuple()

    valid_scores = np.array(
        [s for _, _, s in pairs if not np.isnan(s)], dtype=float
    )
    if valid_scores.size == 0:
        ranks_pct = {idx: 0.0 for idx in range(len(pairs))}
    else:
        order = np.argsort(np.argsort(valid_scores))
        denom = max(valid_scores.size - 1, 1)
        pct_lookup: Dict[float, float] = {}
        for raw, rk in zip(valid_scores, order):
            pct_lookup.setdefault(float(raw), float(rk) / float(denom))
        ranks_pct = {}
        for idx, (_, _, score) in enumerate(pairs):
            ranks_pct[idx] = (
                0.0 if np.isnan(score) else pct_lookup.get(float(score), 0.0)
            )

    edge_views: List[EdgeView] = []
    for idx, (u, v, score) in enumerate(pairs):
        lon_u, lat_u = coords[u]
        lon_v, lat_v = coords[v]
        edge_views.append(
            EdgeView(
                u=u,
                v=v,
                lon_u=lon_u,
                lat_u=lat_u,
                lon_v=lon_v,
                lat_v=lat_v,
                llbce=0.0 if np.isnan(score) else score,
                llbce_rank_pct=ranks_pct[idx],
            )
        )
    return tuple(edge_views)


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------


def _matched_hub(name: str) -> Optional[str]:
    """Return the canonical trunk-hub label whose name occurs in *name* (ci)."""
    if not name:
        return None
    haystack = name.lower()
    for hub in TRUNK_HUBS:
        if hub.lower() in haystack:
            return hub
    return None


# ---------------------------------------------------------------------------
# Folium HTML output
# ---------------------------------------------------------------------------


def render_html(
    node_views: Tuple[NodeView, ...],
    edge_views: Tuple[EdgeView, ...],
    out_path: str,
) -> bool:
    """Render the interactive HTML map; return ``True`` on success."""
    try:
        import folium  # type: ignore
        from branca.colormap import LinearColormap  # type: ignore
    except ImportError:
        print(
            "[render_kanto_map] WARNING: 'folium' (and 'branca') not installed; "
            "skipping HTML output. Install with: pip install folium",
            file=sys.stderr,
        )
        return False

    fmap = folium.Map(
        location=[KANTO_CENTER[0], KANTO_CENTER[1]],
        zoom_start=8,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.map.Marker(
        [KANTO_CENTER[0] + 1.3, KANTO_CENTER[1]],
        icon=folium.DivIcon(
            html=(
                f'<div style="font-size:14pt;font-weight:bold;'
                f'background:white;padding:4px 8px;border-radius:4px;'
                f'box-shadow:1px 1px 4px rgba(0,0,0,.25);">{TITLE}</div>'
            )
        ),
    ).add_to(fmap)

    edge_cmap = LinearColormap(
        colors=["#3b4cc0", "#dddddd", "#b40426"],
        vmin=0.0,
        vmax=1.0,
        caption="LLBCe rank percentile (1.0 = most critical)",
    )

    for ev in edge_views:
        weight = 2.0 + 4.0 * ev.llbce_rank_pct
        opacity = 0.55 + 0.4 * ev.llbce_rank_pct
        folium.PolyLine(
            locations=[(ev.lat_u, ev.lon_u), (ev.lat_v, ev.lon_v)],
            color=edge_cmap(ev.llbce_rank_pct),
            weight=weight,
            opacity=opacity,
            tooltip=(
                f"Edge ({ev.u}, {ev.v}) | LLBCe={ev.llbce:.4g} "
                f"| rank pct={ev.llbce_rank_pct:.2f}"
            ),
        ).add_to(fmap)

    if node_views:
        max_kshell = max(nv.kshell for nv in node_views) or 1
        max_degree = max(nv.degree for nv in node_views) or 1
        node_cmap = LinearColormap(
            colors=["#440154", "#21908d", "#fde725"],
            vmin=0.0,
            vmax=float(max_kshell),
            caption="k-shell index",
        )
        for nv in node_views:
            radius = 3.0 + 8.0 * (nv.degree / max_degree)
            folium.CircleMarker(
                location=(nv.lat, nv.lon),
                radius=radius,
                color=node_cmap(float(nv.kshell)),
                fill=True,
                fill_opacity=0.85,
                weight=1,
                tooltip=(
                    f"{nv.name or '(unnamed)'} "
                    f"| degree={nv.degree} | k-shell={nv.kshell}"
                ),
            ).add_to(fmap)

            hub = _matched_hub(nv.name)
            if hub:
                folium.map.Marker(
                    [nv.lat, nv.lon],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:10pt;font-weight:bold;'
                            f'color:#222;background:rgba(255,255,255,.8);'
                            f'padding:1px 4px;border:1px solid #888;'
                            f'border-radius:3px;white-space:nowrap;">{hub}</div>'
                        )
                    ),
                ).add_to(fmap)

        edge_cmap.add_to(fmap)
        node_cmap.add_to(fmap)

    fmap.save(out_path)
    print(f"[render_kanto_map] Wrote interactive map: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Matplotlib PNG output
# ---------------------------------------------------------------------------


def render_png(
    node_views: Tuple[NodeView, ...],
    edge_views: Tuple[EdgeView, ...],
    out_path: str,
) -> bool:
    """Render the static PNG; return ``True`` on success."""
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.collections import LineCollection  # type: ignore
    except ImportError:
        print(
            "[render_kanto_map] WARNING: matplotlib unavailable; "
            "skipping PNG output.",
            file=sys.stderr,
        )
        return False

    try:
        from pyproj import Transformer  # type: ignore

        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )

        def project(lon: float, lat: float) -> Tuple[float, float]:
            x, y = transformer.transform(lon, lat)
            return float(x), float(y)

        target_crs = "EPSG:3857"
    except ImportError:
        print(
            "[render_kanto_map] WARNING: 'pyproj' not installed; using raw "
            "lon/lat axes (no Web Mercator). Install with: pip install pyproj",
            file=sys.stderr,
        )

        def project(lon: float, lat: float) -> Tuple[float, float]:
            return float(lon), float(lat)

        target_crs = "EPSG:4326"

    try:
        import contextily as cx  # type: ignore

        have_basemap = True
    except ImportError:
        cx = None  # type: ignore[assignment]
        have_basemap = False
        print(
            "[render_kanto_map] WARNING: 'contextily' not installed; rendering "
            "without basemap. Install with: pip install contextily",
            file=sys.stderr,
        )

    fig, ax = plt.subplots(figsize=(12, 11))

    segments: List[List[Tuple[float, float]]] = []
    edge_colors: List[float] = []
    edge_widths: List[float] = []
    for ev in edge_views:
        segments.append([project(ev.lon_u, ev.lat_u), project(ev.lon_v, ev.lat_v)])
        edge_colors.append(ev.llbce_rank_pct)
        edge_widths.append(0.8 + 2.4 * ev.llbce_rank_pct)

    edge_cmap = plt.get_cmap("RdYlBu_r")
    if segments:
        lc = LineCollection(
            segments,
            array=np.asarray(edge_colors),
            cmap=edge_cmap,
            linewidths=edge_widths,
            alpha=0.85,
            zorder=2,
        )
        lc.set_clim(0.0, 1.0)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("LLBCe rank percentile (1.0 = most critical)")

    if node_views:
        xs: List[float] = []
        ys: List[float] = []
        sizes: List[float] = []
        kshells: List[float] = []
        for nv in node_views:
            x, y = project(nv.lon, nv.lat)
            xs.append(x)
            ys.append(y)
            sizes.append(10.0 + 14.0 * nv.degree)
            kshells.append(float(nv.kshell))

        sc = ax.scatter(
            xs,
            ys,
            s=sizes,
            c=kshells,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.95,
            zorder=3,
        )
        cbar2 = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar2.set_label("k-shell index")

        for nv in node_views:
            hub = _matched_hub(nv.name)
            if hub:
                x, y = project(nv.lon, nv.lat)
                ax.annotate(
                    hub,
                    xy=(x, y),
                    xytext=(8, 6),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="#444",
                        alpha=0.85,
                    ),
                    zorder=4,
                )

    ax.set_title(TITLE, fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Easting (m)" if target_crs == "EPSG:3857" else "Longitude")
    ax.set_ylabel("Northing (m)" if target_crs == "EPSG:3857" else "Latitude")

    if have_basemap and target_crs == "EPSG:3857" and segments:
        try:
            cx.add_basemap(  # type: ignore[union-attr]
                ax,
                source=cx.providers.OpenStreetMap.Mapnik,  # type: ignore[union-attr]
                crs=target_crs,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[render_kanto_map] WARNING: contextily basemap failed "
                f"({exc!r}); continuing without basemap.",
                file=sys.stderr,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_kanto_map] Wrote static map: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"[render_kanto_map] Loading graph: {GML_PATH}")
    graph = load_graph(GML_PATH)
    print(
        f"[render_kanto_map] Graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    print(f"[render_kanto_map] Loading node coords: {NODES_CSV}")
    nodes_df = load_node_coords(NODES_CSV)

    print(f"[render_kanto_map] Loading metrics: {METRICS_CSV}")
    metrics = load_metrics(METRICS_CSV)

    node_views, _ = build_node_views(graph, nodes_df)
    edge_views = build_edge_views(graph, metrics, node_views)
    print(
        f"[render_kanto_map] Renderable: {len(node_views)} nodes, "
        f"{len(edge_views)} edges"
    )

    render_html(node_views, edge_views, OUT_HTML)
    render_png(node_views, edge_views, OUT_PNG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
