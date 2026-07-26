"""
Build a nicely formatted static-vs-iterative comparison workbook.

Reads results_comparison/static_vs_iterative_auc.csv (no recompute) and writes
static_vs_iterative_comparison.xlsx with:
  - Overview sheet: per-benchmark "Iterative wins X/6 method groups" + win bars.
  - One color-coded sheet per benchmark: 9 metric rows; the winning AUC cell is
    green ("good"), the losing AUC cell is red ("bad"); Winner column colored too.
  - Cross-Network Summary: mean AUC per metric, static vs iterative, with delta.

Win tally uses 6 method GROUPS (CI's 4 variants collapsed by mean AUC) so the
count reads X/6, while the detail table still shows all 9 metric columns.
Lower AUC = better dismantling.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).parent
CSV = ROOT / "results_comparison" / "static_vs_iterative_auc.csv"
XLSX = ROOT / "static_vs_iterative_comparison.xlsx"

METRIC_ORDER = [
    "LDC", "Jaccard", "LKS",
    "CI_e_av_skin", "CI_e_mul_skin", "CI_e_av_body", "CI_e_mul_body",
    "LLBCe", "LLBMEe1",
]
# Method groups for the X/6 tally (CI's 4 variants collapse to one group).
GROUPS = {
    "LDC": ["LDC"],
    "Jaccard": ["Jaccard"],
    "LKS": ["LKS"],
    "CI": ["CI_e_av_skin", "CI_e_mul_skin", "CI_e_av_body", "CI_e_mul_body"],
    "LLBCe": ["LLBCe"],
    "LLBMEe1": ["LLBMEe1"],
}
N_GROUPS = len(GROUPS)  # 6

# Excel-classic "good"/"bad"/"neutral" palettes.
GOOD_FILL = PatternFill("solid", fgColor="C6EFCE")   # green
GOOD_FONT = Font(color="006100")
BAD_FILL = PatternFill("solid", fgColor="FFC7CE")     # red
BAD_FONT = Font(color="9C0006")
HEAD_FILL = PatternFill("solid", fgColor="1F2A44")    # dark navy
HEAD_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=14, color="1F2A44")
SUB_FONT = Font(bold=True, size=11)
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")


def group_wins(sub: pd.DataFrame) -> tuple[int, int]:
    """Return (iterative_group_wins, n_groups) collapsing CI variants by mean AUC."""
    iwins = 0
    for _g, cols in GROUPS.items():
        part = sub[sub["Metric"].isin(cols)]
        if part.empty:
            continue
        if part["Iterative_AUC"].mean() < part["Static_AUC"].mean():
            iwins += 1
    return iwins, N_GROUPS


def style_header(ws, row: int, headers: list[str]) -> None:
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = CENTER
        cell.border = BORDER


def benchmark_sheet(wb: Workbook, name: str, sub: pd.DataFrame) -> tuple[int, int]:
    ws = wb.create_sheet(name[:31])
    iwins, ng = group_wins(sub)

    ws["A1"] = f"{name} — Static vs Iterative dismantling AUC (lower = better)"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:F1")

    tally = ws["A2"]
    tally.value = f"Iterative wins: {iwins}/{ng} method groups"
    tally.font = SUB_FONT
    tally.fill = GOOD_FILL if iwins > ng / 2 else BAD_FILL
    ws.merge_cells("A2:F2")
    ws["A2"].alignment = LEFT

    headers = ["Metric", "Static AUC", "Iterative AUC", "Δ (I−S)", "% Change", "Winner"]
    style_header(ws, 4, headers)

    sub = sub.set_index("Metric").reindex([m for m in METRIC_ORDER if m in set(sub["Metric"])]).reset_index()
    r = 5
    for _, row in sub.iterrows():
        s, i = float(row["Static_AUC"]), float(row["Iterative_AUC"])
        iter_better = i < s
        delta = i - s
        pct = (delta / s) if s else 0.0
        ws.cell(row=r, column=1, value=row["Metric"]).alignment = LEFT
        c_s = ws.cell(row=r, column=2, value=round(s, 4))
        c_i = ws.cell(row=r, column=3, value=round(i, 4))
        c_d = ws.cell(row=r, column=4, value=round(delta, 4))
        c_p = ws.cell(row=r, column=5, value=pct)
        c_w = ws.cell(row=r, column=6, value="Iterative" if iter_better else "Static")
        # Color winner green ("good"), loser red ("bad").
        if iter_better:
            c_i.fill, c_i.font = GOOD_FILL, GOOD_FONT
            c_s.fill, c_s.font = BAD_FILL, BAD_FONT
            c_w.fill, c_w.font = GOOD_FILL, GOOD_FONT
        else:
            c_s.fill, c_s.font = GOOD_FILL, GOOD_FONT
            c_i.fill, c_i.font = BAD_FILL, BAD_FONT
            c_w.fill, c_w.font = BAD_FILL, BAD_FONT
        c_p.number_format = "0.0%"
        for c in range(1, 7):
            cell = ws.cell(row=r, column=c)
            cell.border = BORDER
            if c >= 2:
                cell.alignment = CENTER
        r += 1

    widths = [16, 13, 14, 11, 11, 12]
    for c, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = "A5"
    return iwins, ng


def overview_sheet(wb: Workbook, df: pd.DataFrame, tallies: dict) -> None:
    ws = wb.create_sheet("Overview", 0)
    ws["A1"] = "Static vs Iterative — Win Summary by Benchmark"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:D1")
    ws["A2"] = "Win = lower dismantling AUC. CI's 4 variants count as one method group (X/6)."
    ws["A2"].font = Font(italic=True, color="666666")
    ws.merge_cells("A2:D2")

    style_header(ws, 4, ["Benchmark", "Iterative wins", "Static wins", "Verdict"])
    r = 5
    tot_i = tot_n = 0
    for name, (iwins, ng) in tallies.items():
        tot_i += iwins
        tot_n += ng
        ws.cell(row=r, column=1, value=name).alignment = LEFT
        ws.cell(row=r, column=2, value=f"{iwins}/{ng}").alignment = CENTER
        ws.cell(row=r, column=3, value=f"{ng - iwins}/{ng}").alignment = CENTER
        verdict = ws.cell(row=r, column=4, value="Iterative" if iwins > ng / 2 else
                          ("Tie" if iwins == ng / 2 else "Static"))
        verdict.alignment = CENTER
        if iwins > ng / 2:
            verdict.fill, verdict.font = GOOD_FILL, GOOD_FONT
            ws.cell(row=r, column=2).fill, ws.cell(row=r, column=2).font = GOOD_FILL, GOOD_FONT
        elif iwins < ng / 2:
            verdict.fill, verdict.font = BAD_FILL, BAD_FONT
            ws.cell(row=r, column=3).fill, ws.cell(row=r, column=3).font = GOOD_FILL, GOOD_FONT
        for c in range(1, 5):
            ws.cell(row=r, column=c).border = BORDER
        r += 1
    # Total row
    ws.cell(row=r, column=1, value="TOTAL").font = SUB_FONT
    tcell = ws.cell(row=r, column=2, value=f"{tot_i}/{tot_n}")
    tcell.font = SUB_FONT
    tcell.fill = GOOD_FILL if tot_i > tot_n / 2 else BAD_FILL
    tcell.alignment = CENTER
    ws.cell(row=r, column=3, value=f"{tot_n - tot_i}/{tot_n}").alignment = CENTER
    for c in range(1, 5):
        ws.cell(row=r, column=c).border = BORDER
    for c, w in enumerate([26, 14, 12, 12], start=1):
        ws.column_dimensions[get_column_letter(c)].width = w


def cross_network_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Cross-Network Summary")
    ws["A1"] = "Mean AUC per metric across all benchmarks (lower = better)"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:E1")
    style_header(ws, 3, ["Metric", "Mean Static", "Mean Iterative", "Δ (I−S)", "Iter win rate"])
    metrics = [m for m in METRIC_ORDER if m in set(df["Metric"])]
    r = 4
    for m in metrics:
        sub = df[df["Metric"] == m]
        ms, mi = sub["Static_AUC"].mean(), sub["Iterative_AUC"].mean()
        wins = int((sub["Iterative_AUC"] < sub["Static_AUC"]).sum())
        n = len(sub)
        ws.cell(row=r, column=1, value=m).alignment = LEFT
        cs = ws.cell(row=r, column=2, value=round(ms, 4))
        ci = ws.cell(row=r, column=3, value=round(mi, 4))
        ws.cell(row=r, column=4, value=round(mi - ms, 4)).alignment = CENTER
        ws.cell(row=r, column=5, value=f"{wins}/{n}").alignment = CENTER
        if mi < ms:
            ci.fill, ci.font = GOOD_FILL, GOOD_FONT
            cs.fill, cs.font = BAD_FILL, BAD_FONT
        else:
            cs.fill, cs.font = GOOD_FILL, GOOD_FONT
            ci.fill, ci.font = BAD_FILL, BAD_FONT
        cs.alignment = ci.alignment = CENTER
        for c in range(1, 6):
            ws.cell(row=r, column=c).border = BORDER
        r += 1
    for c, w in enumerate([16, 13, 14, 11, 13], start=1):
        ws.column_dimensions[get_column_letter(c)].width = w


def main() -> None:
    df = pd.read_csv(CSV)
    wb = Workbook()
    wb.remove(wb.active)

    tallies: dict = {}
    for name in df["Network"].drop_duplicates():
        sub = df[df["Network"] == name]
        tallies[name] = benchmark_sheet(wb, name, sub)

    overview_sheet(wb, df, tallies)          # inserted at index 0
    cross_network_sheet(wb, df)
    try:
        wb.save(XLSX)
        out = XLSX
    except PermissionError:
        out = XLSX.with_name("static_vs_iterative_comparison_formatted.xlsx")
        wb.save(out)
        print(f"NOTE: {XLSX.name} is locked (open in Excel?) — wrote {out.name} instead.")

    print(f"Wrote {out.name}")
    print("Per-benchmark iterative wins (method groups):")
    for name, (iw, ng) in tallies.items():
        print(f"  {name:26s} {iw}/{ng}")


if __name__ == "__main__":
    main()
