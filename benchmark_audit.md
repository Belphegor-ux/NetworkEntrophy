# Benchmark Audit (Karate / Jazz / Football)

Read-only audit of the per-dataset benchmark trees against `new_instructions.md`. Date: 2026-04-27. Branch: `iterative-method`.

Severity: **BLOCKER** = spec violation, **MEDIUM** = legacy/cleanup, **LOW** = cosmetic.

---

## A. EI/IE deprecation (spec §3: "drop the EI method")

The directory `IE/` exists in all three datasets and `rank_ie` lives in `aggregate_results.py:42-63` and is registered in the methods list at `aggregate_results.py:208`.

| File | Status |
|---|---|
| `Karate/IE/IE_informative_entropy.py` | Still active — emits `IE` column |
| `Karate/IE/iterative_ie.py:13` | Imports `rank_ie` |
| `Jazz/IE/IE_informative_entropy.py` | Still active |
| `Jazz/IE/iterative_ie.py` | Still active |
| `Football/IE/IE_informative_entropy.py` | Still active |
| `Football/IE/iterative_ie.py` | Still active |
| `run_karate.sh:8` | Spawns IE worker |
| `run_jazz.sh` | Spawns IE worker |
| `run_football.sh` | Spawns IE worker |
| `aggregate_results.py:42, 208` | Defines and registers `rank_ie` |

**Severity: MEDIUM.** Spec §3 explicitly says drop. Recommendation: keep the directories (historical reproducibility), but (1) remove the `IE` line from `aggregate_results.py:208`, (2) remove the IE invocations from `run_*.sh`, and (3) add a `# DEPRECATED — see new_instructions.md §3` header to each `IE_informative_entropy.py`. Do not delete files (would break checked-in `dashboard_data.json`).

---

## B. LLBCe / LLBMEe1 implementation (spec §4)

Spec mandates `nx.edge_betweenness_centrality_subset` over `Γ(e) = {u, v} ∪ N(u) ∪ N(v)` as both sources and targets. Subgraph extraction is **explicitly forbidden**.

### LLBC/ directories — COMPLIANT

All three use `edge_betweenness_centrality_subset` on the full graph with `fc = {u, v} | N(u) | N(v)`:

- `Karate/LLBC/LLBC_contrast.py:27` — `nx.edge_betweenness_centrality_subset(G, sources=fc_nodes, targets=fc_nodes, normalized=False)` ✓
- `Jazz/LLBC/LLBC_contrast.py:225` — same call (after a long abandoned in-line Brandes attempt; final result uses the subset call) ✓
- `Football/LLBC/LLBC_contrast.py:27` — same ✓

### ME/ directories — BLOCKER (forbidden subgraph extraction)

All three `ME/ME_mapping_entropy.py` files use `G.subgraph(nodes)` + plain `nx.edge_betweenness_centrality(subg, ...)` to compute the LLBC term that feeds LLBME — exactly the pattern §4 forbids.

- `Karate/ME/ME_mapping_entropy.py:18-22` — `subg = G.subgraph(nodes); ebc = nx.edge_betweenness_centrality(subg, ...)` **BLOCKER**
- `Jazz/ME/ME_mapping_entropy.py:18-22` — same **BLOCKER**
- `Football/ME/ME_mapping_entropy.py:18-22` — same **BLOCKER**

Additional concerns in `ME/`:
- Output column is `LLBMEe_improved` (line 60) with ad-hoc `(1 - jaccard)` weighting and `epsilon=1e-6` smoothing — these are not in the spec; spec §4 cites Eq. 11 from MDPI Entropy 26(4) 315 directly.
- The `LLBC/` directory already returns both `LLBC` and `LLBME` correctly. The `ME/` directory is a parallel, divergent implementation that should be reconciled or removed.

**Severity: BLOCKER for spec compliance + MEDIUM duplication.** Recommendation: delete the `ME/` static implementation and have iterative_me.py import from `LLBC/LLBC_contrast.py` (consume the spec-compliant LLBME column there); or rewrite `ME_mapping_entropy.py` to use the subset call.

---

## C. DataFrame schema (spec §1: columns `i, j, <metric>`)

All audited `rank_*` functions return DataFrames with `i, j` first. **No dict/list returns observed.** Compliant across the board:

- `LDC/LDC_link_degree_centrality.py` (all three datasets) → `create_metric_df(..., "LDC")` ✓
- `Jaccard/jaccard_index.py` (all three) → `create_metric_df(..., "Jaccard")` ✓
- `CKS/CKS_Link_K_Shell.py` (all three) → `create_metric_df(..., "LKS")` ✓ (correct column name)
- `IE/IE_informative_entropy.py` (all three) → DataFrame with `IE` ✓ (but spec says drop the metric)
- `LLBC/LLBC_contrast.py` (all three) → DataFrame with `LLBC, LLBME` ✓ schema, but **column names do not match spec** — should be `LLBCe, LLBMEe1` (LOW: rename only).
- `ME/ME_mapping_entropy.py` (all three) → DataFrame with `LLBMEe_improved` ✓ schema (but see B).
- `CI/collective_influence.py` (Karate, Jazz) → DataFrame with 4 CI columns ✓
- `Football/CI/collective_influence.py:43-48` → DataFrame with **only `CI` column** — see D.

---

## D. CI 4-variant requirement (spec §2)

Spec mandates `CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body`.

| File | Columns emitted | Status |
|---|---|---|
| `Karate/CI/collective_influence.py:37-43` | 4 variants | **COMPLIANT** ✓ |
| `Jazz/CI/collective_influence.py:37-43` | 4 variants | **COMPLIANT** ✓ |
| `Football/CI/collective_influence.py:42-48` | 1 column (`CI` = `mul_skin`) | **BLOCKER** |
| `aggregate_results.py:115-135` | 1 column (`CI`) | **BLOCKER** (already noted in `CLAUDE.md` Known Issues #2) |

The Karate/Jazz fix appears completed — Football was missed. Same body computed (line 39) but never emitted.

---

## Summary table

| Dataset | LDC | Jaccard | CKS→LKS | IE | LLBC | ME | CI |
|---|---|---|---|---|---|---|---|
| **Karate** | ✓ | ✓ | ✓ | dep | ✓ (rename to LLBCe) | **BLOCKER** subgraph | ✓ 4-var |
| **Jazz** | ✓ | ✓ | ✓ | dep | ✓ (rename to LLBCe) | **BLOCKER** subgraph | ✓ 4-var |
| **Football** | ✓ | ✓ | ✓ | dep | ✓ (rename to LLBCe) | **BLOCKER** subgraph | **BLOCKER** 1-var |

Compliance ranking (best → worst):
1. **Karate, Jazz** — fully compliant except the `ME/` subgraph violation (B) and column-name LOW (LLBC→LLBCe).
2. **Football** — same `ME/` violation **plus** unfixed CI 1-variant regression.

## Recommended action order
1. Port Karate/Jazz `rank_ci` (4-variant) to `Football/CI/collective_influence.py` and to `aggregate_results.py:rank_ci`. (BLOCKER)
2. Replace all three `ME/ME_mapping_entropy.py` subgraph extraction with `edge_betweenness_centrality_subset`, or delete `ME/` and re-route iterative_me to consume `LLBC/`'s `LLBME` column. (BLOCKER)
3. Rename `LLBC`→`LLBCe`, `LLBME`→`LLBMEe1` in all three `LLBC_contrast.py` files for spec parity. (LOW)
4. Drop IE from `aggregate_results.py:208` and `run_*.sh`; mark IE files DEPRECATED. (MEDIUM)
