"""Rebuild ``static_vs_iterative_comparison.xlsx`` from the current
``dashboard_app/dashboard_data.json``.

Reads RGC curves stored under
``{dataset: {"Static"|"Iterative": {metric: {"x": [...], "y": [...], ...}}}}``,
recomputes AUC via ``np.trapezoid``, applies legacy → spec column-name
renames (``CKS`` → ``LKS``, ``LLBC`` → ``LLBCe``, ``LLBME`` → ``LLBMEe1``),
drops the deprecated IE method, and writes a clean three-sheet workbook
with merged dataset headers (no spacer rows).

To refresh the underlying data, run ``aggregate_results.py`` first.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

DASHBOARD_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dashboard_app",
    "dashboard_data.json",
)
OUT_XLSX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static_vs_iterative_comparison.xlsx",
)

LEGACY_RENAMES: Dict[str, str] = {
    "CKS": "LKS",
    "LLBC": "LLBCe",
    "LLBME": "LLBMEe1",
}
DROPPED_METRICS = {"IE"}

DATASET_ORDER: Tuple[str, ...] = ("Karate", "Football", "Jazz", "Tokyo City")
METRIC_ORDER: Tuple[str, ...] = (
    "LDC",
    "Jaccard",
    "LKS",
    "LLBCe",
    "LLBMEe1",
    "CI",
    "CI_e_av_skin",
    "CI_e_mul_skin",
    "CI_e_av_body",
    "CI_e_mul_body",
)


def _canonical(name: str) -> str:
    """Apply legacy renames; pass through unknown names."""
    return LEGACY_RENAMES.get(name, name)


def _auc(curve: Dict) -> Optional[float]:
    """Return AUC of an RGC curve, or ``None`` if data is missing/invalid."""
    if not isinstance(curve, dict):
        return None
    x = curve.get("x")
    y = curve.get("y")
    if not x or not y or len(x) != len(y):
        if "auc" in curve and isinstance(curve["auc"], (int, float)):
            return float(curve["auc"])
        return None
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(arr_y, arr_x))
    return float(np.trapz(arr_y, arr_x))  # type: ignore[attr-defined]


def _collect_aucs(
    raw: Dict,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """{dataset: {metric_canonical: {"Static": auc|None, "Iterative": auc|None}}}."""
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for ds_name, modes in raw.items():
        out.setdefault(ds_name, {})
        for mode in ("Static", "Iterative"):
            metrics = modes.get(mode, {}) if isinstance(modes, dict) else {}
            for raw_name, curve in metrics.items():
                if raw_name in DROPPED_METRICS:
                    continue
                m = _canonical(raw_name)
                out[ds_name].setdefault(m, {"Static": None, "Iterative": None})
                out[ds_name][m][mode] = _auc(curve)
    return out


def _better(s: Optional[float], i: Optional[float]) -> str:
    if s is None and i is None:
        return "—"
    if i is None:
        return "Static (no iter data)"
    if s is None:
        return "Iterative (no static data)"
    if i < s - 1e-9:
        return "Iterative"
    if s < i - 1e-9:
        return "Static"
    return "Tie"


def _pct_change(s: Optional[float], i: Optional[float]) -> Optional[float]:
    if s is None or i is None or s == 0:
        return None
    return (i - s) / abs(s) * 100.0


def build_workbook(
    auc_map: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    out_path: str,
) -> None:
    try:
        import openpyxl  # type: ignore
        from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
    except ImportError:
        print(
            "ERROR: openpyxl is required. Install with: "
            ".venv/Scripts/python.exe -m pip install openpyxl",
            file=sys.stderr,
        )
        raise

    wb = openpyxl.Workbook()
    header_font = Font(bold=True, size=11)
    title_font = Font(bold=True, size=12, color="FFFFFF")
    title_fill = PatternFill("solid", fgColor="305496")
    iter_fill = PatternFill("solid", fgColor="DEEBF7")
    static_fill = PatternFill("solid", fgColor="FFF2CC")
    center = Alignment(horizontal="center", vertical="center")

    # ----- Sheet 1: Static vs Iterative -----
    ws = wb.active
    ws.title = "Static vs Iterative"
    headers = [
        "Dataset",
        "Metric",
        "Static AUC",
        "Iterative AUC",
        "Difference (I-S)",
        "% Change",
        "Better",
    ]
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.alignment = center

    row_idx = 2
    for ds in DATASET_ORDER:
        ds_data = auc_map.get(ds, {})
        if not ds_data:
            continue
        # Iterate metrics in stable order, then any extras (alphabetised)
        ordered_metrics = [m for m in METRIC_ORDER if m in ds_data]
        ordered_metrics += sorted(set(ds_data) - set(ordered_metrics))
        for m in ordered_metrics:
            s = ds_data[m].get("Static")
            i = ds_data[m].get("Iterative")
            diff = (i - s) if (s is not None and i is not None) else None
            pct = _pct_change(s, i)
            ws.cell(row=row_idx, column=1, value=ds)
            ws.cell(row=row_idx, column=2, value=m)
            ws.cell(row=row_idx, column=3, value=s if s is not None else "—")
            ws.cell(row=row_idx, column=4, value=i if i is not None else "—")
            ws.cell(
                row=row_idx, column=5, value=diff if diff is not None else "—"
            )
            ws.cell(
                row=row_idx,
                column=6,
                value=f"{pct:+.1f}%" if pct is not None else "—",
            )
            ws.cell(row=row_idx, column=7, value=_better(s, i))
            for c in range(3, 6):
                cell = ws.cell(row=row_idx, column=c)
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
            row_idx += 1

    # Column widths
    widths = [12, 18, 12, 14, 14, 10, 26]
    for idx, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = w

    # ----- Sheet 2: Cross-Dataset Summary -----
    ws2 = wb.create_sheet("Cross-Dataset Summary")
    ws2.cell(row=1, column=1, value="Iterative-vs-Static AUC delta per metric").font = title_font
    ws2.cell(row=1, column=1).fill = title_fill
    ws2.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(DATASET_ORDER) + 2)
    ws2.cell(row=1, column=1).alignment = center

    summary_headers = ["Metric"] + list(DATASET_ORDER) + ["Iterative win rate"]
    for col_idx, h in enumerate(summary_headers, start=1):
        cell = ws2.cell(row=2, column=col_idx, value=h)
        cell.font = header_font
        cell.alignment = center

    # Collect union of metrics across datasets
    metric_union: List[str] = []
    seen = set()
    for ds in DATASET_ORDER:
        for m in auc_map.get(ds, {}):
            if m not in seen:
                metric_union.append(m)
                seen.add(m)
    metric_union.sort(
        key=lambda m: METRIC_ORDER.index(m) if m in METRIC_ORDER else 999
    )

    row_idx = 3
    for m in metric_union:
        ws2.cell(row=row_idx, column=1, value=m).font = header_font
        wins = 0
        compared = 0
        for col_offset, ds in enumerate(DATASET_ORDER, start=2):
            ds_data = auc_map.get(ds, {}).get(m, {})
            s = ds_data.get("Static")
            i = ds_data.get("Iterative")
            if s is None or i is None:
                ws2.cell(row=row_idx, column=col_offset, value="—")
                continue
            delta = i - s
            cell = ws2.cell(row=row_idx, column=col_offset, value=delta)
            cell.number_format = "+0.0000;-0.0000"
            if delta < 0:
                cell.fill = iter_fill
                wins += 1
            elif delta > 0:
                cell.fill = static_fill
            compared += 1
        ws2.cell(
            row=row_idx,
            column=len(summary_headers),
            value=f"{wins}/{compared}" if compared else "—",
        )
        row_idx += 1

    widths2 = [16] + [14] * len(DATASET_ORDER) + [18]
    for idx, w in enumerate(widths2, start=1):
        ws2.column_dimensions[get_column_letter(idx)].width = w

    # ----- Sheet 3: Static-Only Rankings -----
    ws3 = wb.create_sheet("Static-Only Rankings")
    ws3.cell(
        row=1, column=1, value="Static AUC ranking per dataset (lower = better)"
    ).font = title_font
    ws3.cell(row=1, column=1).fill = title_fill
    ws3.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)
    ws3.cell(row=1, column=1).alignment = center

    headers3 = ["Dataset", "Rank", "Metric", "Static AUC"]
    for col_idx, h in enumerate(headers3, start=1):
        cell = ws3.cell(row=2, column=col_idx, value=h)
        cell.font = header_font
        cell.alignment = center

    row_idx = 3
    for ds in DATASET_ORDER:
        ds_data = auc_map.get(ds, {})
        scored = [(m, vals.get("Static")) for m, vals in ds_data.items()]
        scored = [(m, s) for m, s in scored if s is not None]
        scored.sort(key=lambda t: t[1])
        for rank, (m, s) in enumerate(scored, start=1):
            ws3.cell(row=row_idx, column=1, value=ds)
            ws3.cell(row=row_idx, column=2, value=rank)
            ws3.cell(row=row_idx, column=3, value=m)
            cell = ws3.cell(row=row_idx, column=4, value=s)
            cell.number_format = "0.0000"
            row_idx += 1

    widths3 = [14, 8, 22, 14]
    for idx, w in enumerate(widths3, start=1):
        ws3.column_dimensions[get_column_letter(idx)].width = w

    wb.save(out_path)


def main() -> int:
    if not os.path.exists(DASHBOARD_JSON):
        print(
            f"ERROR: {DASHBOARD_JSON} not found. "
            f"Run aggregate_results.py first.",
            file=sys.stderr,
        )
        return 1
    with open(DASHBOARD_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    auc_map = _collect_aucs(raw)
    n_metrics = sum(len(v) for v in auc_map.values())
    print(
        f"[build_xlsx] Loaded AUCs for {len(auc_map)} dataset(s), "
        f"{n_metrics} (dataset, metric) pairs."
    )

    build_workbook(auc_map, OUT_XLSX)
    print(f"[build_xlsx] Wrote: {OUT_XLSX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
