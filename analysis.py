#!/usr/bin/env python3
"""Stage 2 - every number, table and statistical test reported in the paper.

Inputs: the per-subject CSVs written by pipeline.py (15 ms run, and optionally the
35 ms robustness run) and the BIDS metadata of ds004080 (SOZ, resection and anatomical
labels). The channel-noise tests also read the first 120 s of raw data, unless the
cache _channel_noise.pkl is present in --out-dir.

Outputs, in --results-dir: one CSV per table or test, and report.txt, which lists
every reported value next to the place where it appears in the paper.

Usage:
    python analysis.py --bids-root data/ds004080 --out-dir outputs/art15 \
                       --out-dir-35 outputs/art35 --results-dir results
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

import pipeline as P


# =============================================================================
# PARAMETERS (every random step has its own fixed seed)
# =============================================================================

TFZ_ORDER = ["TN1", "TN2", "AN1", "AN2", "BN1", "BN2",
             "GlN1", "GlN2", "GhN1", "GhN2", "WR"]
MAIN_ZONE = "GhN2"                      # pre-specified zone of the original article

X_PERCENT = np.linspace(0.1, 10.0, 100)  # top X % of pairs, as in the article
X_CLINICAL = 2.0                         # clinically relevant regime: X <= 2 %
N_ITER = 100                             # surrogate draws per subject and zone
PERCENTILE = 95.0
RESAMPLE_PER_X = False    # True: new surrogate matrix for every X, as in the original
                          # MATLAB code; False: one matrix per draw, evaluated at all X
SEED_SURROGATES = 20260811

N_BOOT = 4000                            # bootstrap over subjects (margin over all X)
SEED_BOOT = 20260817

FLAGGED = ["ccepAgeUMCU07", "ccepAgeUMCU37", "ccepAgeUMCU44", "ccepAgeUMCU52",
           "ccepAgeUMCU53", "ccepAgeUMCU59", "ccepAgeUMCU60", "ccepAgeUMCU63"]
# flagged at visual inspection of the mean evoked potentials: 07 and 60 saturating
# drift, 37 and 44 no identifiable response, 52 permanent ~1 mV oscillation,
# 53 pre-stimulus contamination, 59 very low signal-to-noise ratio, 63 late transient
N_PERM_FLAGGED = 500
SEED_FLAGGED = 20260816

NOISE_SECONDS = 120.0                    # raw data read for the channel-noise index
SEED_NOISE = 20260816

N_PERM_ANATOMY = 1000
SEED_ANATOMY = 20260817
TEMPORAL_KEYS = ("temp", "hippocamp", "parahip", "amygdal", "fusiform",
                 "collat", "planum", "heschl")    # Destrieux labels of the temporal lobe

TRIALS_THRESHOLD = 8                     # strata: median trials per site < 8 or >= 8
CROSS_ZONES = ("GhN2", "GhN1")           # zones of the crossover statistic
N_PERM_STRATA = 1000
SEED_STRATA = 20260817


def x_index(x):
    return int(np.argmin(np.abs(X_PERCENT - x)))


# =============================================================================
# LOCALISATION METRICS (funct_spec_sens.m, funct_rand_spec_sens.m)
# =============================================================================

def pair_contacts(df):
    """The four monopolar contacts (stim 1, stim 2, resp 1, resp 2) of every pair."""
    s = df["stim"].astype(str).map(P.parse_site)
    r = df["resp"].astype(str).map(P.parse_site)
    return (np.array([a for a, _ in s]), np.array([b for _, b in s]),
            np.array([a for a, _ in r]), np.array([b for _, b in r]))


def spec_sens_curves(values, s1, s2, r1, r2, labels, x_percent=X_PERCENT):
    """Specificity and sensitivity (%) of the top X % of pairs ranked by `values`.

    Specificity: share of the ceil(N * X / 100) top pairs with at least one of their
    four contacts labelled. Sensitivity: share of the labelled contacts appearing
    among these pairs, as stimulated or recorded contact.
    """
    ok = np.isfinite(values)
    v, n = values[ok], int(ok.sum())
    if n == 0 or not labels:
        return np.full(len(x_percent), np.nan), np.full(len(x_percent), np.nan)

    order = np.argsort(-v, kind="stable")
    a1, a2, b1, b2 = s1[ok][order], s2[ok][order], r1[ok][order], r2[ok][order]
    is_target = np.array([(x in labels) or (y in labels) or (z in labels) or (w in labels)
                          for x, y, z, w in zip(a1, a2, b1, b2)])
    cum = np.cumsum(is_target)

    spec = np.empty(len(x_percent))
    sens = np.empty(len(x_percent))
    for i, xp in enumerate(x_percent):
        k = max(min(int(np.ceil(n * xp / 100.0)), n), 1)
        spec[i] = 100.0 * cum[k - 1] / k
        seen = set(a1[:k]) | set(a2[:k]) | set(b1[:k]) | set(b2[:k])
        sens[i] = 100.0 * len(labels & seen) / len(labels)
    return spec, sens


def surrogate_curves(raw_values, weights, s1, s2, r1, r2, labels, rng):
    """N_ITER surrogate curves: the UNWEIGHTED fractions are resampled with
    replacement, the volume-conduction weights are re-applied, and the curves are
    recomputed."""
    ok = np.isfinite(raw_values)
    pool = raw_values[ok]
    S = np.full((N_ITER, len(X_PERCENT)), np.nan)
    E = np.full((N_ITER, len(X_PERCENT)), np.nan)
    if pool.size == 0 or not labels:
        return S, E
    for it in range(N_ITER):
        if RESAMPLE_PER_X:
            for i, x in enumerate(X_PERCENT):
                draw = np.full(raw_values.shape, np.nan)
                draw[ok] = rng.choice(pool, size=int(ok.sum()), replace=True)
                sp, se = spec_sens_curves(draw * weights, s1, s2, r1, r2, labels, [x])
                S[it, i], E[it, i] = sp[0], se[0]
        else:
            draw = np.full(raw_values.shape, np.nan)
            draw[ok] = rng.choice(pool, size=int(ok.sum()), replace=True)
            S[it], E[it] = spec_sens_curves(draw * weights, s1, s2, r1, r2, labels)
    return S, E


def analyse_subject(csv_path, labels, rng):
    """Observed and surrogate curves of one subject, for every zone."""
    subject = os.path.basename(csv_path).split("_significance")[0].replace("sub-", "")
    if not labels:
        return None
    df = pd.read_csv(csv_path)
    s1, s2, r1, r2 = pair_contacts(df)
    weights = df["weight"].values.astype(float)
    reachable = labels & (set(s1) | set(s2) | set(r1) | set(r2))
    out = dict(subject=subject, n_pairs=len(df), n_soz=len(labels),
               n_soz_reachable=len(reachable), true={}, surr={}, surr95={})
    for tfz in TFZ_ORDER:
        if tfz not in df.columns:
            continue
        raw = df[tfz].values.astype(float)
        wv = df[tfz + "_w"].values.astype(float) if tfz + "_w" in df.columns else raw * weights
        sp, se = spec_sens_curves(wv, s1, s2, r1, r2, labels)
        S, E = surrogate_curves(raw, weights, s1, s2, r1, r2, labels, rng)
        out["true"][tfz] = dict(spec=sp, sens=se)
        out["surr"][tfz] = dict(spec=S, sens=E)
        out["surr95"][tfz] = dict(spec=np.nanpercentile(S, PERCENTILE, axis=0),
                                  sens=np.nanpercentile(E, PERCENTILE, axis=0))
    return out


def subject_results(out_dir, bids_root, target="soz", force=False):
    """analyse_subject for every CSV of out_dir (sorted), cached on disk.

    target: 'soz' (hypothesised seizure onset zone) or 'resected'.
    """
    cache = os.path.join(out_dir, "_subject_results.pkl" if target == "soz"
                         else "_resected_results.pkl")
    if os.path.exists(cache) and not force:
        with open(cache, "rb") as f:
            return pickle.load(f)
    rng = np.random.default_rng(SEED_SURROGATES)
    subs = []
    for csv in sorted(glob.glob(os.path.join(out_dir, "*_significance_fractions.csv"))):
        subject = os.path.basename(csv).split("_significance")[0].replace("sub-", "")
        labels = P.labelled_contacts(P.electrodes_table(bids_root, subject), target)
        s = analyse_subject(csv, labels, rng)
        if s:
            subs.append(s)
    with open(cache, "wb") as f:
        pickle.dump(subs, f)
    return subs


def group_analysis(subs):
    """Group criterion of the article: the median across subjects is significant at X
    when it exceeds the 95th percentile of the distribution of surrogate medians."""
    res = {}
    for tfz in TFZ_ORDER:
        grp = [s for s in subs if tfz in s["true"]]
        if not grp:
            continue
        r = dict(n=len(grp), x=X_PERCENT)
        for key in ("spec", "sens"):
            true = np.vstack([s["true"][tfz][key] for s in grp])
            n_iter = grp[0]["surr"][tfz][key].shape[0]
            med = np.vstack([np.nanmedian(np.vstack([s["surr"][tfz][key][it] for s in grp]),
                                          axis=0) for it in range(n_iter)])
            r[f"{key}_true"] = np.nanmedian(true, axis=0)
            r[f"{key}_rand"] = np.nanmedian(med, axis=0)
            r[f"{key}_95"] = np.nanpercentile(med, PERCENTILE, axis=0)
            r[f"{key}_indiv"] = true
            r[f"{key}_sig"] = r[f"{key}_true"] > r[f"{key}_95"]
        res[tfz] = r
    return res


# =============================================================================
# SUBJECT-LEVEL MARGIN, BOOTSTRAP AND BENJAMINI-HOCHBERG
# =============================================================================

def build_arrays(subs):
    """Observed (subjects x X) and surrogate (subjects x draws x X) specificity."""
    zones = [z for z in TFZ_ORDER if all(z in s["true"] for s in subs)]
    T = {z: np.vstack([s["true"][z]["spec"] for s in subs]) for z in zones}
    U = {z: np.stack([s["surr"][z]["spec"] for s in subs], axis=0) for z in zones}
    return T, U, zones


def group_margin(T, U, zone, idx, x_sel=None):
    """Mean over X (all X, or the X selected by x_sel) of: median across subjects idx
    of the specificity, minus the median over draws of the median across the same
    subjects of the surrogate specificity."""
    tm = np.nanmedian(T[zone][idx], axis=0)
    sm = np.nanmedian(np.nanmedian(U[zone][idx], axis=0), axis=0)
    d = tm - sm
    return float(np.nanmean(d if x_sel is None else d[x_sel]))


def bootstrap_margins(subs, target):
    """Margin averaged over all X (0.1-10 %), 95 % percentile CI and two-sided p from
    N_BOOT resamples of subjects, then Benjamini-Hochberg over the 11 zones (p floored
    at 1 / N_BOOT)."""
    T, U, zones = build_arrays(subs)
    x_sel = None
    rng = np.random.default_rng(SEED_BOOT)
    n, rows = len(subs), []
    for z in zones:
        b = np.array([group_margin(T, U, z, rng.choice(n, n, replace=True), x_sel)
                      for _ in range(N_BOOT)])
        p = min(max(2 * min(np.mean(b <= 0), np.mean(b >= 0)), 1.0 / N_BOOT), 1.0)
        rows.append(dict(target=target, zone=z, n_subjects=n,
                         margin=round(float(np.mean(b)), 1),
                         ci_low=round(float(np.percentile(b, 2.5)), 1),
                         ci_high=round(float(np.percentile(b, 97.5)), 1),
                         p=round(float(p), 4)))
    d = pd.DataFrame(rows)
    rej, q = fdrcorrection(d["p"].values, alpha=0.05, method="indep")
    d["q_bh"] = np.round(q, 4)
    d["significant_after_bh"] = rej
    return d


def per_subject_table(subs, tfz=MAIN_ZONE):
    """Each subject's specificity against the 95th percentile of its own surrogates;
    mean gap over X <= 2 % > +5 points: informative, < -5: counter-informative."""
    sel = X_PERCENT <= X_CLINICAL
    rows = []
    for s in subs:
        r = dict(subject=s["subject"], n_pairs=s["n_pairs"], n_soz=s["n_soz"])
        for x in (0.1, 1.0, 5.0):
            i = x_index(x)
            r[f"spec_at_{x:g}"] = round(float(s["true"][tfz]["spec"][i]), 1)
            r[f"random95_at_{x:g}"] = round(float(s["surr95"][tfz]["spec"][i]), 1)
            r[f"sens_at_{x:g}"] = round(float(s["true"][tfz]["sens"][i]), 1)
        gap = s["true"][tfz]["spec"][sel] - s["surr95"][tfz]["spec"][sel]
        r["mean_gap_x2"] = round(float(np.nanmean(gap)), 1)
        r["class"] = ("informative" if r["mean_gap_x2"] > 5 else
                      "borderline" if r["mean_gap_x2"] > -5 else "counter-informative")
        r["flagged"] = s["subject"] in FLAGGED
        rows.append(r)
    return pd.DataFrame(rows).sort_values("mean_gap_x2", ascending=False)


# =============================================================================
# CHANNEL NOISE
# =============================================================================

def channel_noise(subject, bids_root):
    """Standard deviation (uV) of every bipolar channel over the first 120 s of the
    first run."""
    ses, run = P.find_runs(bids_root, subject, P.CFG["task"])[0]
    raw, df_elec, _, _ = P.load_run(bids_root, subject, ses, run, P.CFG["task"])
    raw.crop(tmax=min(NOISE_SECONDS, raw.times[-1]))
    bip, names, _, _ = P.build_bipolar(raw, df_elec)
    return dict(zip(names, np.std(bip, axis=1) * 1e6))


def all_channel_noise(subjects, out_dir, bids_root):
    cache = os.path.join(out_dir, "_channel_noise.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    noise = {s: channel_noise(s, bids_root) for s in subjects}
    with open(cache, "wb") as f:
        pickle.dump(noise, f)
    return noise


def noise_soz_ratio(subs, noise, bids_root):
    """Median noise of SOZ channels / median noise of the other channels, per subject;
    Wilcoxon signed-rank test of (ratio - 1), two-sided."""
    rows = []
    for s in subs:
        sub = s["subject"]
        if sub not in noise:
            continue
        soz = P.labelled_contacts(P.electrodes_table(bids_root, sub), "soz")
        if not soz:
            continue
        n_soz, n_other = [], []
        for chan, v in noise[sub].items():
            a, b = P.parse_site(str(chan))
            (n_soz if (a in soz or b in soz) else n_other).append(v)
        if len(n_soz) < 2 or len(n_other) < 5:
            continue
        rows.append(dict(subject=sub, ratio=round(float(np.median(n_soz) /
                                                        max(np.median(n_other), 1e-9)), 3)))
    df = pd.DataFrame(rows)
    return df, wilcoxon(df["ratio"] - 1.0)


def _curves_for_values(df, values, labels):
    s1, s2, r1, r2 = pair_contacts(df)
    v = np.asarray(values, dtype=float) * df["weight"].values.astype(float)
    return spec_sens_curves(v, s1, s2, r1, r2, labels)


def _residualise(y, x):
    """Residual of y after a linear regression on log(x); NaN where undefined."""
    ok = np.isfinite(y) & np.isfinite(x) & (x > 0)
    out = np.full_like(y, np.nan, dtype=float)
    if ok.sum() < 10:
        return out
    A = np.vstack([np.log(x[ok]), np.ones(int(ok.sum()))]).T
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    out[ok] = y[ok] - (A @ coef)
    return out


def noise_competitors(subs, noise, out_dir, bids_root, tfz=MAIN_ZONE):
    """Pairs ranked by the GhN2 fraction, by the noise of the response channel alone,
    and by the GhN2 fraction residualised on that noise: mean margin over X <= 2 %
    above the 95th percentile of the subject's own surrogates."""
    sel = X_PERCENT <= X_CLINICAL
    rows = []
    for s in subs:
        sub = s["subject"]
        csv = os.path.join(out_dir, f"sub-{sub}_significance_fractions.csv")
        if sub not in noise or not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        soz = P.labelled_contacts(P.electrodes_table(bids_root, sub), "soz")
        if not soz:
            continue
        frac = df[tfz].values.astype(float)
        nz = df["resp"].astype(str).map(lambda c: noise[sub].get(c, np.nan)).values.astype(float)

        sp_g, _ = _curves_for_values(df, frac, soz)
        sp_n, _ = _curves_for_values(df, nz, soz)
        sp_r, _ = _curves_for_values(df, _residualise(frac, nz), soz)
        rng = np.random.default_rng(SEED_NOISE)
        S, _ = surrogate_curves(frac, df["weight"].values.astype(float),
                                *pair_contacts(df), soz, rng)
        ref = np.nanpercentile(S, PERCENTILE, axis=0)
        rows.append(dict(subject=sub,
                         margin_ghn2=round(float(np.nanmean(sp_g[sel] - ref[sel])), 1),
                         margin_noise=round(float(np.nanmean(sp_n[sel] - ref[sel])), 1),
                         margin_residual=round(float(np.nanmean(sp_r[sel] - ref[sel])), 1),
                         spearman_ghn2_noise=round(float(spearmanr(
                             frac, nz, nan_policy="omit").statistic), 3)))
    return pd.DataFrame(rows)


# =============================================================================
# FLAGGED RECORDINGS, ANATOMY
# =============================================================================

def flagged_removal(subs, tfz=MAIN_ZONE):
    """Group result for X <= 2 % without the flagged recordings, against N_PERM_FLAGGED
    removals of the same number of random subjects (one-sided p)."""
    sel = X_PERCENT <= X_CLINICAL
    k = sum(1 for f in FLAGGED if f in [s["subject"] for s in subs])

    def score(subset):
        g = group_analysis(subset)[tfz]
        return (100.0 * float(g["spec_sig"][sel].mean()),
                float(np.nanmean(g["spec_true"][sel] - g["spec_rand"][sel])))

    full = score(subs)
    obs = score([s for s in subs if s["subject"] not in FLAGGED])
    rng = np.random.default_rng(SEED_FLAGGED)
    perm = []
    for _ in range(N_PERM_FLAGGED):
        idx = rng.choice(len(subs), size=len(subs) - k, replace=False)
        perm.append(score([subs[j] for j in idx]))
    perm = np.array(perm)
    return dict(k=k, full_pct=full[0], without_flagged_pct=obs[0],
                p=float(np.mean(perm[:, 0] <= obs[0]))), perm


def classify_anatomy(subjects, bids_root):
    """Temporal / extratemporal by the majority of the Destrieux labels of SOZ contacts."""
    rows = []
    for sub in subjects:
        df = P.electrodes_table(bids_root, sub)
        col = next((c for c in ("Destrieux_label_text", "Destrieux_label")
                    if c in df.columns), None)
        labs = []
        if col is not None and "soz" in df.columns:
            m = df["soz"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"])
            labs = [x for x in df.loc[m, col].astype(str).str.strip()
                    if x and x.lower() not in ("n/a", "nan", "none", "0")]
        n_t = sum(1 for lab in labs if any(k in lab.lower() for k in TEMPORAL_KEYS))
        n_e = len(labs) - n_t
        group = ("undetermined" if not labs else "temporal" if n_t > n_e else
                 "extratemporal" if n_e > n_t else "mixed")
        rows.append(dict(subject=sub, n_soz_labelled=len(labs), n_temporal=n_t,
                         n_extratemporal=n_e, group=group))
    return pd.DataFrame(rows)


def anatomy_test(subs, cls, tfz=MAIN_ZONE):
    """Margin over X <= 2 %, temporal minus extratemporal; two-sided label permutation
    test."""
    T, U, _ = build_arrays(subs)
    x_sel = X_PERCENT <= X_CLINICAL
    grp = cls.set_index("subject")["group"].to_dict()
    lab = np.array([grp.get(s["subject"], "undetermined") for s in subs])
    idx_t = np.where(lab == "temporal")[0]
    idx_e = np.where(lab == "extratemporal")[0]
    m_t = group_margin(T, U, tfz, idx_t, x_sel)
    m_e = group_margin(T, U, tfz, idx_e, x_sel)
    pool = np.concatenate([idx_t, idx_e])
    rng = np.random.default_rng(SEED_ANATOMY)
    perm = np.empty(N_PERM_ANATOMY)
    for i in range(N_PERM_ANATOMY):
        sh = rng.permutation(pool)
        perm[i] = (group_margin(T, U, tfz, sh[:len(idx_t)], x_sel)
                   - group_margin(T, U, tfz, sh[len(idx_t):], x_sel))
    obs = m_t - m_e
    return dict(n_temporal=len(idx_t), n_extratemporal=len(idx_e),
                margin_temporal=round(m_t, 1), margin_extratemporal=round(m_e, 1),
                difference=round(obs, 1),
                p=round(float(np.mean(np.abs(perm) >= abs(obs))), 4)), perm


# =============================================================================
# TRIAL-COUNT STRATIFICATION (exploratory)
# =============================================================================

def trials_per_subject(subs, out_dir):
    """Median number of trials per site of each subject (10, or 5 per polarity)."""
    return pd.Series({s["subject"]: float(pd.read_csv(
        os.path.join(out_dir, f"sub-{s['subject']}_significance_fractions.csv"),
        usecols=["n_trials"])["n_trials"].median()) for s in subs})


def trial_count_stratification(subs, trials):
    """Margin over X <= 2 % in subjects with fewer than 8 trials versus the others.
    Stratum labels are permuted N_PERM_STRATA times (group sizes kept) to test, per
    zone, the margin difference (Benjamini-Hochberg over zones) and the crossover
    (GhN2 difference - GhN1 difference). The omnibus p compares the observed crossover
    with the maximum crossover over the 55 zone pairs of each permutation, which
    corrects for having selected this pair after inspection."""
    T, U, zones = build_arrays(subs)
    sel = X_PERCENT <= X_CLINICAL
    is_lo = np.array([trials[s["subject"]] < TRIALS_THRESHOLD for s in subs])
    idx_lo, idx_hi = np.where(is_lo)[0], np.where(~is_lo)[0]

    def diff(z, lo, hi):
        return group_margin(T, U, z, lo, sel) - group_margin(T, U, z, hi, sel)

    obs = {z: diff(z, idx_lo, idx_hi) for z in zones}
    rng = np.random.default_rng(SEED_STRATA)
    perm = {z: np.empty(N_PERM_STRATA) for z in zones}
    all_idx = np.arange(len(subs))
    for i in range(N_PERM_STRATA):
        lo = rng.choice(all_idx, size=int(is_lo.sum()), replace=False)
        hi = np.setdiff1d(all_idx, lo)
        for z in zones:
            perm[z][i] = diff(z, lo, hi)

    table = pd.DataFrame([dict(zone=z, n_low=len(idx_lo), n_high=len(idx_hi),
                               margin_low=round(group_margin(T, U, z, idx_lo, sel), 1),
                               margin_high=round(group_margin(T, U, z, idx_hi, sel), 1),
                               difference=round(obs[z], 1),
                               p=round(float(np.mean(np.abs(perm[z]) >= abs(obs[z]))), 4))
                          for z in zones])
    table["q_bh"] = np.round(fdrcorrection(table["p"].values, alpha=0.05,
                                           method="indep")[1], 4)

    za, zb = CROSS_ZONES
    cross = obs[za] - obs[zb]
    P_ = np.array([perm[z] for z in zones])
    iu = np.triu_indices(len(zones), k=1)                   # the 55 zone pairs
    max_cross = np.abs(P_[:, None, :] - P_[None, :, :])[iu].max(axis=0)
    return table, dict(crossover=round(cross, 1),
                       p_crossover=float(np.mean(np.abs(perm[za] - perm[zb]) >= abs(cross))),
                       p_omnibus=float(np.mean(max_cross >= abs(cross))))


def competing_splits(subs, trials, noise):
    """GhN2 and GhN1 margin differences for other median splits of the subjects
    (trials split at 8): is the trial count the only variable giving this divide?"""
    T, U, _ = build_arrays(subs)
    sel = X_PERCENT <= X_CLINICAL
    meta = pd.DataFrame(dict(
        n_trials_median=[trials[s["subject"]] for s in subs],
        n_pairs=[s["n_pairs"] for s in subs],
        index=[int("".join(c for c in s["subject"] if c.isdigit())) for s in subs],
        n_soz=[s["n_soz"] for s in subs],
        noise_median=[round(float(np.median(list(noise[s["subject"]].values()))), 1)
                      for s in subs]))
    za, zb = CROSS_ZONES
    rows = []
    for var in meta.columns:
        thr = TRIALS_THRESHOLD if var == "n_trials_median" else meta[var].median()
        lo = np.where(meta[var].values < thr)[0]
        hi = np.where(meta[var].values >= thr)[0]
        if len(lo) < 5 or len(hi) < 5:
            continue
        d_a = group_margin(T, U, za, lo, sel) - group_margin(T, U, za, hi, sel)
        d_b = group_margin(T, U, zb, lo, sel) - group_margin(T, U, zb, hi, sel)
        rows.append(dict(variable=var, threshold=round(float(thr), 1), n_low=len(lo),
                         n_high=len(hi), difference_ghn2=round(d_a, 1),
                         difference_ghn1=round(d_b, 1), crossover=round(d_a - d_b, 1)))
    return pd.DataFrame(rows).sort_values("crossover", key=abs, ascending=False)


# =============================================================================
# COHORT DESCRIPTION
# =============================================================================

def stimulated_fraction(bids_root, subject):
    """Share of implanted contacts (grid, strip, depth) that were stimulated."""
    ieeg = os.path.join(bids_root, f"sub-{subject}", "ses-*", "ieeg")
    el = pd.concat([pd.read_csv(f, sep="\t")
                    for f in sorted(glob.glob(os.path.join(ieeg, "*_electrodes.tsv")))])
    implanted = set(el.loc[el["group"].isin(["grid", "strip", "depth"]), "name"].astype(str))
    stim = set()
    for f in sorted(glob.glob(os.path.join(ieeg, "*_events.tsv"))):
        ev = pd.read_csv(f, sep="\t")
        ev = ev[ev["trial_type"] == "electrical_stimulation"]
        for site in ev["electrical_stimulation_site"].astype(str):
            stim |= {c.strip() for c in site.split("-")}
    return len(stim & implanted) / len(implanted)


def cohort(out_dir, bids_root):
    selected, n_soz = P.select_subjects(bids_root)
    csvs = sorted(glob.glob(os.path.join(out_dir, "*_significance_fractions.csv")))
    tables = {os.path.basename(c).split("_significance")[0][4:]: pd.read_csv(c) for c in csvs}
    subjects = list(tables)
    n_pairs = pd.Series({s: len(t) for s, t in tables.items()})
    trials = pd.Series({s: float(t["n_trials"].median()) for s, t in tables.items()})
    dist = pd.concat([t["dist_mm"] for t in tables.values()])
    n_labels = {tgt: pd.Series({s: len(P.labelled_contacts(P.electrodes_table(bids_root, s),
                                                           tgt)) for s in subjects})
                for tgt in ("soz", "resected")}
    stim = pd.Series({s: stimulated_fraction(bids_root, s) for s in subjects})
    first10 = pd.concat([tables[s] for s in P.SUBJECTS[:10] if s in tables])

    out = dict(
        n_subjects_with_1_soz=int(sum(v >= 1 for v in n_soz.values())),
        n_subjects_with_3_soz=len(selected),
        selection_matches_list=sorted(selected) == sorted(P.SUBJECTS),
        stimulated_contacts_pct=(round(100 * stim.min()), round(100 * stim.max())),
        n_subjects=len(subjects), n_pairs_total=int(n_pairs.sum()),
        n_pairs_per_subject=(int(n_pairs.min()), int(n_pairs.max())),
        n_subjects_10_trials=int((trials == 10).sum()),
        n_subjects_5_trials=int((trials == 5).sum()),
        soz_contacts=(int(n_labels["soz"].min()), float(n_labels["soz"].median()),
                      int(n_labels["soz"].max())),
        resected_contacts=(int(n_labels["resected"][n_labels["resected"] > 0].min()),
                           float(n_labels["resected"][n_labels["resected"] > 0].median()),
                           int(n_labels["resected"].max())),
        n_subjects_without_resection=int((n_labels["resected"] == 0).sum()),
        distance_mm_min_median=(round(float(dist.min()), 1), round(float(dist.median()), 1)),
        volume_conduction_first10=dict(n_pairs=len(first10),
                                       ghn2_weighted=round(float(first10["GhN2_w"].mean()), 3),
                                       ghn2_unweighted=round(float(first10["GhN2"].mean()), 3)),
    )
    summaries = [os.path.join(out_dir, f"sub-{s}_summary.json") for s in subjects]
    if all(os.path.exists(f) for f in summaries):      # written by pipeline.py
        ret = []
        for f in summaries:
            with open(f) as fh:
                d = json.load(fh)
            ret.append(100.0 * d["n_pairs_kept"] / d["n_pairs_evaluated"])
        out["retention_pct"] = (round(min(ret), 1), round(float(np.median(ret)), 1),
                                round(max(ret), 1))
    return out


# =============================================================================
# MAIN
# =============================================================================

def zone_significance(res, sel):
    """Percentage of X values (all, or X <= 2 %) where the group median is significant."""
    return {tfz: (100.0 * float(res[tfz]["spec_sig"].mean()),
                  100.0 * float(res[tfz]["spec_sig"][sel].mean()),
                  100.0 * float(res[tfz]["sens_sig"].mean()))
            for tfz in TFZ_ORDER if tfz in res}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--out-dir", required=True, help="CSVs of the 15 ms run")
    ap.add_argument("--out-dir-35", help="CSVs of the 35 ms robustness run")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    rd = lambda name: os.path.join(args.results_dir, name)    # noqa: E731
    lines = []
    say = lambda s="": (print(s), lines.append(s))            # noqa: E731
    sel = X_PERCENT <= X_CLINICAL

    # --- Methods and Results > Cohort ---------------------------------------
    c = cohort(args.out_dir, args.bids_root)
    with open(rd("cohort.json"), "w") as f:
        json.dump(c, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    say("COHORT (Methods, Results > Cohort)")
    for k, v in c.items():
        say(f"  {k}: {v}")

    # --- Results > Replication of the group-level result; Table 2 ----------
    subs = subject_results(args.out_dir, args.bids_root, "soz")
    res = group_analysis(subs)
    rows = []
    for tfz, (all_x, x2, sens) in zone_significance(res, sel).items():
        row = dict(zone=tfz, spec_pct_all_x=round(all_x, 1), spec_pct_x2=round(x2, 1),
                   sens_pct_all_x=round(sens, 1), first_significant_x=None,
                   pairs_at_first_x_min=None, pairs_at_first_x_max=None)
        sig_x = X_PERCENT[res[tfz]["spec_sig"]]
        if sig_x.size:       # number of top pairs at the first significant X
            n_top = [max(min(int(np.ceil(s["n_pairs"] * sig_x[0] / 100.0)), s["n_pairs"]), 1)
                     for s in subs]
            row.update(first_significant_x=round(float(sig_x[0]), 1),
                       pairs_at_first_x_min=min(n_top), pairs_at_first_x_max=max(n_top))
        rows.append(row)
    table2 = pd.DataFrame(rows)
    table2.to_csv(rd("table2_group_significance.csv"), index=False)
    say("\nGROUP CRITERION OF THE ARTICLE (Results, Table 2, Figure 2A)")
    say(table2.to_string(index=False))

    # --- Results > No zone survives subject-level uncertainty; Figure 3 ----
    subs_res = subject_results(args.out_dir, args.bids_root, "resected")
    boot = pd.concat([bootstrap_margins(subs, "hypothesised SOZ"),
                      bootstrap_margins(subs_res, "resected tissue")], ignore_index=True)
    boot.to_csv(rd("subject_level_bootstrap.csv"), index=False)
    say(f"\nSUBJECT-LEVEL BOOTSTRAP, {N_BOOT} resamples, margin averaged over all X "
        f"(Abstract, Results, Figure 3)")
    say(boot.to_string(index=False))

    # --- Results > A cohort split in two; Figure 4 --------------------------
    tab = per_subject_table(subs)
    tab.to_csv(rd("per_subject_ghn2.csv"), index=False)
    say("\nPER-SUBJECT CLASSIFICATION, GhN2, X <= 2 % (Results, Figure 4)")
    say(f"  {tab['class'].value_counts().to_dict()}")

    # --- Results > Influence of artefacts; Figure 5A ------------------------
    noise = all_channel_noise([s["subject"] for s in subs], args.out_dir, args.bids_root)
    d1, w = noise_soz_ratio(subs, noise, args.bids_root)
    d1.to_csv(rd("noise_soz_ratio.csv"), index=False)
    say("\nNOISE OF SOZ CHANNELS (Results > Influence of artefacts)")
    say(f"  median ratio {d1['ratio'].median():.3f}, higher in {(d1['ratio'] > 1).sum()}"
        f"/{len(d1)} subjects, Wilcoxon W = {w.statistic:g}, p = {w.pvalue:.4f}")
    comp = noise_competitors(subs, noise, args.out_dir, args.bids_root)
    comp.to_csv(rd("noise_competitors.csv"), index=False)
    say("  median margins over X <= 2 % (Figure 5A): "
        f"GhN2 {comp['margin_ghn2'].median():+.1f}, noise only "
        f"{comp['margin_noise'].median():+.1f}, residualised "
        f"{comp['margin_residual'].median():+.1f}; median Spearman GhN2-noise "
        f"{comp['spearman_ghn2_noise'].median():+.3f}")

    if args.out_dir_35:
        res35 = group_analysis(subject_results(args.out_dir_35, args.bids_root, "soz"))
        art = pd.DataFrame([dict(zone=t, pct_all_x_15ms=round(a, 1), pct_all_x_35ms=round(b, 1),
                                 pct_x2_15ms=round(a2, 1), pct_x2_35ms=round(b2, 1))
                            for (t, (a, a2, _)), (_, (b, b2, _)) in
                            zip(zone_significance(res, sel).items(),
                                zone_significance(res35, sel).items())])
        art.to_csv(rd("artifact_window_35ms.csv"), index=False)
        say("\nARTEFACT WINDOW 15 -> 35 ms (Results, Discussion)")
        say(art.to_string(index=False))

    flag, perm_flag = flagged_removal(subs)
    pd.DataFrame(perm_flag, columns=["pct_x2", "mean_gap"]).to_csv(
        rd("flagged_random_removals.csv"), index=False)
    say(f"\nFLAGGED RECORDINGS: GhN2 X <= 2 % {flag['full_pct']:.0f} % -> "
        f"{flag['without_flagged_pct']:.0f} % without the {flag['k']} flagged subjects, "
        f"p = {flag['p']:.3f} (one-sided, {N_PERM_FLAGGED} random removals)")

    # --- Results > Exploratory observations; Figure 5B ----------------------
    cls = classify_anatomy([s["subject"] for s in subs], args.bids_root)
    cls.to_csv(rd("anatomy_classification.csv"), index=False)
    anat, perm_anat = anatomy_test(subs, cls)
    pd.DataFrame([anat]).to_csv(rd("anatomy_test.csv"), index=False)
    pd.DataFrame(dict(difference=perm_anat)).to_csv(rd("anatomy_permutations.csv"),
                                                    index=False)
    say(f"\nANATOMY (Figure 5B): {anat}")

    trials = trials_per_subject(subs, args.out_dir)
    strata, cross = trial_count_stratification(subs, trials)
    strata.to_csv(rd("trial_strata_permutation.csv"), index=False)
    splits = competing_splits(subs, trials, noise)
    splits.to_csv(rd("trial_strata_competing_splits.csv"), index=False)
    say(f"\nTRIAL-COUNT STRATIFICATION, X <= 2 %, {N_PERM_STRATA} permutations "
        f"(Results > Exploratory observations)")
    say(strata.to_string(index=False))
    say(f"  crossover {CROSS_ZONES[0]} - {CROSS_ZONES[1]}: {cross['crossover']:+.1f} points, "
        f"p = {cross['p_crossover']:.3f}; omnibus over the 55 zone pairs: "
        f"p = {cross['p_omnibus']:.3f}")
    say("  other splits of the subjects:")
    say(splits.to_string(index=False))

    with open(rd("report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
