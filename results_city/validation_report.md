# Validation Report for Tokyo Power Grid Metrics
## Column Check: ✅ PASSED
All expected columns are present.
## Missing Values Check: ✅ PASSED
No missing values (NaNs) found.
## Data Range & Type Checks
✅ Jaccard: All values within [0, 1]
✅ LDC: All values are non-negative
✅ LKS: All values are non-negative
✅ CI_e_av_skin: All values are non-negative
✅ CI_e_mul_skin: All values are non-negative
✅ CI_e_av_body: All values are non-negative
✅ CI_e_mul_body: All values are non-negative
✅ LLBCe: All values are non-negative
✅ LLBMEe1: all values <= 0 (smallest = most critical)
✅ Edges: All edges are distinct node pairs (i != j)
## Metrics Summary Statistics
```
                i           j         LDC     Jaccard         LKS  CI_e_av_skin  CI_e_mul_skin  CI_e_av_body  CI_e_mul_body       LLBCe     LLBMEe1
count  190.000000  190.000000  190.000000  190.000000  190.000000    190.000000     190.000000    190.000000     190.000000  190.000000  190.000000
mean    47.900000   85.410526    8.952632    0.022107    3.278947     32.463158    1477.278947     71.015789    6979.289474    0.245208  -77.426143
std     38.286521   42.398336    5.883604    0.064133    1.164319     28.296918    2494.367530     61.876924   12209.462749    0.063339  100.280286
min      0.000000    1.000000    2.000000    0.000000    1.000000      0.500000       0.000000      1.500000       0.000000    0.083333 -878.234258
25%     15.250000   52.250000    4.000000    0.000000    2.000000      8.125000      24.000000     23.625000     193.500000    0.196429 -106.200630
50%     41.000000   87.500000    8.000000    0.000000    4.000000     22.750000     376.000000     48.000000    1767.000000    0.250000  -37.943030
75%     73.000000  122.000000   12.000000    0.000000    4.000000     54.375000    1737.000000    108.375000    7488.000000    0.300000  -13.908806
max    148.000000  155.000000   36.000000    0.500000    4.000000    120.000000   14175.000000    272.500000   72450.000000    0.333333   -2.772589
```
