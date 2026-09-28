# External validation of a single-pulse stimulation spectral biomarker

Code for the manuscript *External validation of a single-pulse stimulation spectral
biomarker: from group to patient*.

The cortico-cortical spectral response (CCSR) pipeline of Brinyark et al. (2026),
*Clin Neurophysiol* 186:2111855
([original MATLAB code](https://github.com/UAB-NSPM-Lab/Optimization-of-CCSR)), is
re-implemented in Python. It is applied to the 34 subjects of OpenNeuro ds004080 with at
least three contacts labelled as seizure onset zone (SOZ).

| File | Role |
|---|---|
| `pipeline.py` | Stage 1: raw BIDS data → per-subject significance fractions (one CSV per subject). |
| `analysis.py` | Stage 2: every number, table and statistical test of the paper. |
| `figures.py` | Stage 3: figure panels. |

## Installation

Python 3.11, then:

```
pip install -r requirements.txt
```

## Data

OpenNeuro ds004080, version 1.2.4 (doi:10.18112/openneuro.ds004080.v1.2.4), licence CC0.
Only the 34 subjects listed in `pipeline.SUBJECTS` are needed. For example, with DataLad:

```
datalad install -s https://github.com/OpenNeuroDatasets/ds004080.git data/ds004080
git -C data/ds004080 checkout 1.2.4
python -c "import pipeline; print(' '.join('data/ds004080/sub-' + s for s in pipeline.SUBJECTS))" \
    | xargs datalad get -d data/ds004080
```

## Reproduction

```
python pipeline.py --bids-root data/ds004080 --out-dir outputs/art15
python pipeline.py --bids-root data/ds004080 --out-dir outputs/art35 --artifact-ms 35
python analysis.py --bids-root data/ds004080 --out-dir outputs/art15 --out-dir-35 outputs/art35 --results-dir results
python figures.py  --bids-root data/ds004080 --out-dir outputs/art15 --results-dir results --fig-dir figures
```

- **Stage 1** takes several hours. Each subject runs in its own process, and subjects already computed are skipped.
- **Stages 2 and 3** take minutes. They only need the CSVs of stage 1 and the BIDS metadata (`*.tsv`, `*.json`). The exceptions are the channel-noise index (the first 120 s of raw data, cached in `_channel_noise.pkl`) and Figure 1A–C.
- **Random seeds.** Every random step has its own fixed seed (see the top of `analysis.py`), so all results are exactly reproducible.

## Where each result comes from

`results/report.txt` lists every value below with its place in the paper.

| Paper | Output |
|---|---|
| Subject selection (36 subjects with at least one SOZ contact, 34 with at least three); share of contacts stimulated; pairs, SOZ and resected contacts, distances, volume-conduction check | `results/cohort.json` |
| Retention rate per subject (needs the `*_summary.json` files written by stage 1) | `results/cohort.json` |
| Group criterion of the original article, Table 2, first significant X per zone | `results/table2_group_significance.csv` |
| Subject-level margins averaged over all X (0.1–10 %), 95 % CI, p and Benjamini-Hochberg q, two targets (Abstract, Results, Figure 3) | `results/subject_level_bootstrap.csv` |
| Informative / borderline / counter-informative subjects (Results, Figure 4) | `results/per_subject_ghn2.csv` |
| Noise of SOZ channels (median ratio, Wilcoxon test) | `results/noise_soz_ratio.csv` |
| Ranking by noise alone, residualised ranking, Spearman correlation (Figure 5A) | `results/noise_competitors.csv` |
| Artefact window extended from 15 to 35 ms | `results/artifact_window_35ms.csv` |
| Removal of the eight flagged recordings versus 500 random removals | `results/flagged_random_removals.csv` |
| Temporal versus extratemporal SOZ, margin over X ≤ 2 %, permutation test (Figure 5B) | `results/anatomy_*.csv` |
| Fewer than 8 versus at least 8 trials per site: margins over X ≤ 2 %, permutation tests per zone, GhN2–GhN1 crossover and omnibus over the 55 zone pairs; other splits of the subjects | `results/trial_strata_*.csv` |
| Figures 1 to 5 | `figures/*.png` |

## Deviations from the original implementation

The changes from the original code are listed below.

- **Benjamini-Hochberg correction** covers the whole epoch, baseline included. The original applies it to 15–500 ms.
- **Surrogate matrix.** One matrix is drawn per iteration and evaluated at every X, instead of a new matrix for each X. Set `RESAMPLE_PER_X = True` in `analysis.py` to use the original behaviour.
- **Rejection criteria.** The three criteria described in the article are applied. The four additional amplitude criteria of the original code, calibrated on its hardware, are disabled (`strict_artifact` in `pipeline.py`).
- **Artefact latency.** It is measured on the mean response: the last sample, within 60 ms, that deviates by more than 20 baseline standard deviations.
- **Decimation.** Signals are decimated from 2048 to 1024 Hz before the wavelet transform. Apart from the anti-aliasing filter of the decimation, no software filter is applied, as in the original script. All trials are kept.

## Licence

Code: MIT (see `LICENSE`). Data: CC0, OpenNeuro ds004080.
