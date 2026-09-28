#!/usr/bin/env python3
"""Stage 1 - significance fractions of the cortico-cortical spectral response (CCSR).

Python re-implementation of the pipeline of Brinyark et al. (2026), Clin Neurophysiol
186:2111855 (MATLAB code: https://github.com/UAB-NSPM-Lab/Optimization-of-CCSR),
applied to OpenNeuro ds004080 v1.2.4.

For every stimulation-response pair: bipolar montage, epochs from -500 to +900 ms,
rejection criteria, replacement of the stimulation artefact, decimation to 1024 Hz,
Morse wavelet power in dB, per-pixel test against the baseline with Benjamini-Hochberg
correction, and fraction of significant pixels in 11 time-frequency zones (TFZ).

Output, per subject, in --out-dir:
    sub-<id>_significance_fractions.csv   one row per retained pair
    sub-<id>_summary.json                 pairs evaluated / retained

Usage:
    python pipeline.py --bids-root data/ds004080 --out-dir outputs/art15
    python pipeline.py --bids-root data/ds004080 --out-dir outputs/art35 --artifact-ms 35
"""

import argparse
import gc
import glob
import json
import os
import subprocess
import sys

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from scipy.signal import resample_poly
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PARAMETERS
# =============================================================================

# The 34 subjects of ds004080 with at least three contacts labelled soz == yes
# (see select_subjects(); 36 subjects have at least one).
SUBJECTS = [
    "ccepAgeUMCU07", "ccepAgeUMCU15", "ccepAgeUMCU17", "ccepAgeUMCU21",
    "ccepAgeUMCU26", "ccepAgeUMCU28", "ccepAgeUMCU29", "ccepAgeUMCU31",
    "ccepAgeUMCU33", "ccepAgeUMCU35", "ccepAgeUMCU36", "ccepAgeUMCU37",
    "ccepAgeUMCU38", "ccepAgeUMCU39", "ccepAgeUMCU41", "ccepAgeUMCU42",
    "ccepAgeUMCU44", "ccepAgeUMCU45", "ccepAgeUMCU46", "ccepAgeUMCU47",
    "ccepAgeUMCU48", "ccepAgeUMCU51", "ccepAgeUMCU52", "ccepAgeUMCU53",
    "ccepAgeUMCU55", "ccepAgeUMCU57", "ccepAgeUMCU58", "ccepAgeUMCU59",
    "ccepAgeUMCU60", "ccepAgeUMCU61", "ccepAgeUMCU62", "ccepAgeUMCU63",
    "ccepAgeUMCU65", "ccepAgeUMCU69",
]

CFG = dict(
    task="SPESclin",
    min_soz_contacts=3,

    # epochs and baselines
    tmin=-0.500,
    tmax=0.900,
    baseline=(-0.450, -0.150),       # baseline of the statistical test
    # (the dB normalisation uses 50-450 ms from epoch start, i.e. -450 to -50 ms)

    # stimulation artefact
    artifact_ms=15.0,                # replaced window: -2 to +15 ms
    artifact_pre_ms=2.0,
    max_artifact_latency_ms=15.0,    # rejection criterion (2) of the article

    # rejection criterion (3) of the article
    std_threshold_uv=1000.0,
    bad_trial_fraction=1.0 / 3.0,
    # the four additional criteria of the original code (800/1500/200/200 uV),
    # calibrated on the original hardware, are disabled
    strict_artifact=False,

    # volume conduction weights (weight 0 below 10 mm, 1 above 20 mm)
    dist_min_mm=10.0,
    dist_max_mm=20.0,
    max_pair_dist_mm=25.0,           # adjacent contacts further apart are not paired

    # generalised Morse wavelet, as MATLAB cwt
    fmin=4.0,
    fmax=250.0,
    voices_per_octave=10,
    morse_gamma=3.0,
    morse_time_bandwidth=30.0,

    target_sfreq=1024.0,             # decimation from 2048 Hz

    # statistics
    alpha=0.05,
    pval_window="all",               # the test covers the whole epoch, as in funct_pvals.m
)

FREQ_BANDS = {"T": (4, 8), "A": (8, 13), "B": (13, 30), "Gl": (30, 50), "Gh": (50, 250)}
TIME_BANDS = {"N1": (0.015, 0.050), "N2": (0.050, 0.500)}
WR_ZONE = ((4, 250), (0.015, 0.500))


# =============================================================================
# BIDS INPUT
# =============================================================================

def find_runs(bids_root, subject, task):
    """(session, run) of every run of this subject and task, sorted."""
    pattern = os.path.join(bids_root, f"sub-{subject}", "ses-*", "ieeg",
                           f"sub-{subject}_ses-*_task-{task}_run-*_ieeg.vhdr")
    out = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        ses = base.split("_ses-")[1].split("_")[0]
        run = base.split("_run-")[1].split("_")[0]
        out.append((ses, run))
    return out


def load_run(bids_root, subject, ses, run, task):
    """Raw iEEG (good channels only) and the electrodes, channels and events tables."""
    bp = BIDSPath(subject=subject, session=ses, task=task, run=run,
                  datatype="ieeg", root=bids_root)
    raw = read_raw_bids(bids_path=bp, verbose=False)
    raw.load_data(verbose=False)

    p_elec = bp.copy().update(task=None, run=None, suffix="electrodes",
                              extension=".tsv", check=False)
    df_elec = pd.read_csv(p_elec.fpath, sep="\t")
    p_chan = bp.copy().update(suffix="channels", extension=".tsv", check=False)
    df_chan = pd.read_csv(p_chan.fpath, sep="\t")
    p_evt = bp.copy().update(suffix="events", extension=".tsv", check=False)
    df_evt = pd.read_csv(p_evt.fpath, sep="\t")

    if "status" in df_chan.columns:
        bads = df_chan.loc[df_chan["status"].astype(str).str.lower() == "bad", "name"].tolist()
        raw.info["bads"] = [b for b in bads if b in raw.ch_names]
        raw.drop_channels(raw.info["bads"])

    picks = mne.pick_types(raw.info, ecog=True, seeg=True, eeg=False, misc=False)
    if len(picks) == 0:
        picks = mne.pick_types(raw.info, eeg=True)
    raw.pick(picks)
    return raw, df_elec, df_chan, df_evt


def electrodes_table(bids_root, subject, task=CFG["task"]):
    """electrodes.tsv of the session of the first run (the table read by load_run).
    Located from the channels.tsv files, so that the raw signals are not needed."""
    pattern = os.path.join(bids_root, f"sub-{subject}", "ses-*", "ieeg",
                           f"sub-{subject}_ses-*_task-{task}_run-*_channels.tsv")
    ses = os.path.basename(sorted(glob.glob(pattern))[0]).split("_ses-")[1].split("_")[0]
    path = os.path.join(bids_root, f"sub-{subject}", f"ses-{ses}", "ieeg",
                        f"sub-{subject}_ses-{ses}_electrodes.tsv")
    return pd.read_csv(path, sep="\t")


def labelled_contacts(df_elec, column):
    """Names of the monopolar contacts labelled yes in a column (soz, resected)."""
    if column not in df_elec.columns:
        return set()
    m = df_elec[column].astype(str).str.strip().str.lower().isin(["yes", "1", "true"])
    return set(df_elec.loc[m, "name"].astype(str))


def select_subjects(bids_root, min_soz=CFG["min_soz_contacts"]):
    """Subjects with at least min_soz SOZ contacts, pooled over all electrodes.tsv."""
    n_soz = {}
    for sub_dir in sorted(glob.glob(os.path.join(bids_root, "sub-*"))):
        subject = os.path.basename(sub_dir)[4:]
        soz = set()
        for path in sorted(glob.glob(os.path.join(sub_dir, "ses-*", "ieeg",
                                                  "*_electrodes.tsv"))):
            soz |= labelled_contacts(pd.read_csv(path, sep="\t"), "soz")
        n_soz[subject] = len(soz)
    return [s for s, n in n_soz.items() if n >= min_soz], n_soz


# =============================================================================
# BIPOLAR MONTAGE AND STIMULATION EVENTS
# =============================================================================

def split_name(name):
    """'C12' -> ('C', 12)."""
    i = len(name)
    while i > 0 and name[i - 1].isdigit():
        i -= 1
    if i == len(name):
        return name, None
    return name[:i], int(name[i:])


def build_bipolar(raw, df_elec, max_pair_dist_mm=CFG["max_pair_dist_mm"]):
    """Adjacent contacts of the same electrode (n, n+1), subtracted.

    The position of a bipolar channel is the midpoint of its two contacts. Pairs
    further apart than max_pair_dist_mm (numbering that wraps around a grid) are
    rejected.
    """
    coords = {}
    if {"name", "x", "y", "z"}.issubset(df_elec.columns):
        for _, r in df_elec.iterrows():
            v = pd.to_numeric(pd.Series([r["x"], r["y"], r["z"]]),
                              errors="coerce").values.astype(np.float64)
            if np.all(np.isfinite(v)):
                coords[str(r["name"])] = v

    ch_names = list(raw.ch_names)
    ch_index = {c: i for i, c in enumerate(ch_names)}
    groups = {}
    for c in ch_names:
        pre, num = split_name(c)
        if num is not None:
            groups.setdefault(pre, []).append((num, c))

    pairs = []
    for pre, items in groups.items():
        items.sort()
        for (n1, c1), (n2, c2) in zip(items[:-1], items[1:]):
            if n2 != n1 + 1:
                continue
            if c1 in coords and c2 in coords:
                d = np.linalg.norm(coords[c1] - coords[c2])
                if not np.isfinite(d) or d > max_pair_dist_mm:
                    continue
                pos = 0.5 * (coords[c1] + coords[c2])
            else:
                pos = np.array([np.nan, np.nan, np.nan])
            pairs.append((f"{c1}-{c2}", c1, c2, pos))
    if not pairs:
        raise RuntimeError("no bipolar pair could be built")

    data = raw.get_data()                      # volts
    bip = np.empty((len(pairs), data.shape[1]), dtype=np.float32)
    for k, (_, c1, c2, _) in enumerate(pairs):
        bip[k] = (data[ch_index[c1]] - data[ch_index[c2]]).astype(np.float32)
    del data

    names = [p[0] for p in pairs]
    members = {p[0]: (p[1], p[2]) for p in pairs}
    positions = np.vstack([p[3] for p in pairs])
    return bip, names, members, positions


def get_stim_events(df_evt):
    """Stimulation onsets from events.tsv, grouped by stimulated site."""
    df = df_evt.copy()
    if "trial_type" in df.columns:
        df = df[df["trial_type"].astype(str).str.lower().str.contains("electrical_stimulation")]
    site_col = None
    for c in ("electrical_stimulation_site", "electrical_stimulation_site_name", "site_name"):
        if c in df.columns:
            site_col = c
            break
    if site_col is None:
        raise RuntimeError("no stimulation-site column in events.tsv")
    df = df[df[site_col].notna()]
    df = df[~df[site_col].astype(str).str.strip().str.lower().isin(["n/a", "nan", ""])]
    return {str(site): grp["onset"].astype(float).values
            for site, grp in df.groupby(site_col)}


def parse_site(site):
    """'C1-C2' -> ('C1', 'C2')."""
    for sep in ("-", "_"):
        if sep in site:
            a, b = site.split(sep)[:2]
            return a.strip(), b.strip()
    return site.strip(), None


# =============================================================================
# EPOCHS, REJECTION CRITERIA, ARTEFACT
# =============================================================================

def make_epochs(bip, sfreq, onsets, tmin, tmax):
    """(n_trials, n_channels, n_times) epochs and their time vector."""
    i0 = int(round(tmin * sfreq))
    i1 = int(round(tmax * sfreq))
    n_times = i1 - i0 + 1
    starts = [int(round(on * sfreq)) + i0 for on in onsets]
    starts = [s for s in starts if s >= 0 and s + n_times <= bip.shape[1]]
    if not starts:
        return None, None
    ep = np.stack([bip[:, s:s + n_times] for s in starts], axis=0)
    return ep, np.arange(i0, i1 + 1) / sfreq


def detect_artifact_latency(trial_avg, times, sfreq, baseline_slice, search_ms=60.0):
    """Artefact duration (ms): last sample within search_ms after the stimulus that
    deviates by more than 20 baseline standard deviations."""
    base = trial_avg[baseline_slice]
    sd = np.std(base)
    if sd <= 0:
        return np.inf
    i_stim = int(np.argmin(np.abs(times)))
    i_end = min(i_stim + int(round(search_ms * 1e-3 * sfreq)), len(times))
    seg = np.abs(trial_avg[i_stim:i_end] - np.mean(base))
    above = np.where(seg > 20.0 * sd)[0]
    if len(above) == 0:
        return 0.0
    return (above[-1] + 1) / sfreq * 1000.0


def remove_artifact(ep, times, sfreq, artifact_ms, pre_ms):
    """Replace [-pre_ms, +artifact_ms] by a cross-fade of the time-reversed segments
    preceding the onset and following the artefact (removestimart_getmeanresponse.m)."""
    i_stim = int(np.argmin(np.abs(times)))
    i0 = i_stim - int(round(pre_ms * 1e-3 * sfreq))
    i1 = i_stim + int(round(artifact_ms * 1e-3 * sfreq))
    L = i1 - i0
    if L <= 0 or i0 - L < 0 or i1 + L >= ep.shape[-1]:
        return ep
    pre = ep[..., i0 - L:i0][..., ::-1].copy()
    post = ep[..., i1:i1 + L][..., ::-1].copy()
    w = np.linspace(1.0, 0.0, L)
    ep = ep.copy()
    ep[..., i0:i1] = w * pre + (1.0 - w) * post
    return ep


def screen_pair(ep_ch, times, sfreq, cfg, baseline_slice):
    """Rejection criteria (2) and (3) of the article, on one stimulation-response pair.
    Criterion (1), shared contact, is applied in process_subject. Returns (ok, reason)."""
    mean_uv = ep_ch.mean(axis=0) * 1e6
    n = len(times)

    # (3) standard deviation > 1000 uV on more than one third of the trials
    stds_uv = np.std(ep_ch, axis=-1) * 1e6
    if np.mean(stds_uv > cfg["std_threshold_uv"]) > cfg["bad_trial_fraction"]:
        return False, "std"

    # (2) artefact latency > 15 ms, measured on the mean response
    if detect_artifact_latency(mean_uv, times, sfreq, baseline_slice) \
            > cfg["max_artifact_latency_ms"]:
        return False, "latency"

    if cfg.get("strict_artifact", False):
        # the four additional criteria of funct_removeartifacts.m (disabled here)
        i_a0 = int(np.searchsorted(times, times[0] + 0.050))
        i_a1 = int(np.searchsorted(times, times[0] + 0.450))
        i_b0 = n - int(round(0.450 * sfreq))
        i_b1 = n - int(round(0.050 * sfreq))
        if abs(np.mean(mean_uv[i_b0:i_b1]) - np.mean(mean_uv[i_a0:i_a1])) > 200.0:
            return False, "mean drift"
        if np.max(np.abs(mean_uv)) > 1500.0:
            return False, "residual"
        std_t = np.std(ep_ch, axis=0) * 1e6
        if np.median(std_t) > 800.0:
            return False, "median std"
        if abs(np.mean(std_t[i_b0:i_b1]) - np.mean(std_t[i_a0:i_a1])) > 200.0:
            return False, "std drift"
    return True, ""


# =============================================================================
# WAVELET POWER AND SIGNIFICANCE FRACTIONS
# =============================================================================

def make_freqs(fmin, fmax, voices_per_octave):
    """Logarithmic frequency grid, voices_per_octave points per octave."""
    n = int(np.floor(np.log2(fmax / fmin) * voices_per_octave)) + 1
    return np.logspace(np.log10(fmin), np.log10(fmax), num=n)


def morse_cwt_power(x, sfreq, freqs, gamma=3.0, time_bandwidth=30.0):
    """|W|^2 of a generalised Morse wavelet transform, psi(w) = 2 w^beta exp(-w^gamma),
    beta = time_bandwidth / gamma, peak-normalised at each scale. x: (..., n_times);
    returns (..., n_freqs, n_times) in float32. Trials are processed one at a time
    to bound memory."""
    beta = time_bandwidth / gamma
    x = np.asarray(x, dtype=np.float32)
    if x.ndim >= 2:
        return np.stack([morse_cwt_power(x[i], sfreq, freqs, gamma, time_bandwidth)
                         for i in range(x.shape[0])], axis=0)

    n_times = x.shape[-1]
    n_fft = int(2 ** np.ceil(np.log2(n_times * 2)))
    X = np.fft.fft(x, n=n_fft).astype(np.complex64)
    w = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=1.0)
    w_peak = (beta / gamma) ** (1.0 / gamma)
    scales = w_peak * sfreq / (2.0 * np.pi * np.asarray(freqs, dtype=np.float64))
    pos = w > 0
    power = np.empty((len(freqs), n_times), dtype=np.float32)
    for k, a in enumerate(scales):
        psi = np.zeros(n_fft, dtype=np.float32)
        aw = a * w[pos]
        with np.errstate(over="ignore", under="ignore"):
            psi[pos] = (2.0 * np.exp(beta * np.log(aw) - aw ** gamma)).astype(np.float32)
        mx = float(np.max(psi)) if psi.size else 0.0
        if mx > 0.0:
            psi /= mx
        W = np.fft.ifft(X * psi)[:n_times]
        power[k, :] = (np.abs(W) ** 2).astype(np.float32)
    return power


def significance_mask(ccsr_db, times, freqs, cfg):
    """Significant pixels (funct_pvals.m): p = normcdf of each pixel under a normal fit
    of the baseline at the same frequency, folded above 0.975 (p = 1 - cdf); then
    Benjamini-Hochberg over all tested pixels."""
    b0, b1 = np.searchsorted(times, cfg["baseline"])
    n_f, n_t = ccsr_db.shape
    test_cols = np.where(times > 0)[0] if cfg["pval_window"] == "post" else np.arange(n_t)

    p = np.ones((n_f, n_t))
    for f in range(n_f):
        mu, sd = norm.fit(ccsr_db[f, b0:b1])
        cdf = norm.cdf(ccsr_db[f, test_cols], loc=mu, scale=max(sd, 1e-12))
        p[f, test_cols] = np.where(cdf >= 0.975, 1.0 - cdf, cdf)

    rejected, _ = fdrcorrection(p[:, test_cols].ravel(), alpha=cfg["alpha"], method="indep")
    mask = np.zeros((n_f, n_t), dtype=bool)
    mask[:, test_cols] = rejected.reshape(n_f, len(test_cols))
    return mask


def tfz_slices(freqs, times):
    """Frequency and time selectors of the 11 zones (10 band x window zones + WR)."""
    zones = {}
    for fname, (f0, f1) in FREQ_BANDS.items():
        fsel = (freqs >= f0) & (freqs < f1) if f1 < 250 else (freqs >= f0) & (freqs <= f1)
        for tname, (t0, t1) in TIME_BANDS.items():
            zones[f"{fname}{tname}"] = (fsel, (times >= t0) & (times < t1))
    (f0, f1), (t0, t1) = WR_ZONE
    zones["WR"] = ((freqs >= f0) & (freqs <= f1), (times >= t0) & (times < t1))
    return zones


def significance_fractions(mask, zones):
    """Fraction of significant pixels in each zone."""
    out = {}
    for name, (fsel, tsel) in zones.items():
        sub = mask[np.ix_(fsel, tsel)]
        out[name] = float(sub.mean()) if sub.size else np.nan
    return out


def volume_weight(dist_mm, dmin, dmax):
    """0 below dmin, 1 above dmax, quadratic in between (section 2.5 of the article)."""
    if not np.isfinite(dist_mm):
        return np.nan
    if dist_mm <= dmin:
        return 0.0
    if dist_mm >= dmax:
        return 1.0
    return (dist_mm ** 2 - dmin ** 2) / (dmax ** 2 - dmin ** 2)


def ccsr_db(ep_clean, times, sfreq_orig, freqs, cfg):
    """Decimate, wavelet power, per-trial dB relative to 50-450 ms from epoch start,
    mean over trials. Returns (ccsr in dB, time vector, sampling rate)."""
    sfreq, t_used = sfreq_orig, times
    if cfg["target_sfreq"] and sfreq_orig > cfg["target_sfreq"]:
        q = int(round(sfreq_orig / cfg["target_sfreq"]))
        if q > 1:
            ep_clean = resample_poly(ep_clean, up=1, down=q, axis=-1)
            sfreq = sfreq_orig / q
            t_used = times[0] + np.arange(ep_clean.shape[-1]) / sfreq
    pw = morse_cwt_power(ep_clean, sfreq, freqs, gamma=cfg["morse_gamma"],
                         time_bandwidth=cfg["morse_time_bandwidth"])
    nb0 = int(np.searchsorted(t_used, t_used[0] + 0.050))
    nb1 = int(np.searchsorted(t_used, t_used[0] + 0.450))
    blmp = pw[..., nb0:nb1].mean(axis=-1, keepdims=True)
    db = 10.0 * np.log10(pw / (blmp + 1e-30) + 1e-30)
    return db.mean(axis=0), t_used, sfreq


# =============================================================================
# ONE SUBJECT
# =============================================================================

def process_subject(subject, bids_root, cfg=CFG):
    """All retained stimulation-response pairs of one subject, with their fractions."""
    freqs = make_freqs(cfg["fmin"], cfg["fmax"], cfg["voices_per_octave"])
    zones, records = None, []
    n_total = n_kept = n_excl_dist = 0

    for ses, run in find_runs(bids_root, subject, cfg["task"]):
        raw, df_elec, _, df_evt = load_run(bids_root, subject, ses, run, cfg["task"])
        sfreq_orig = float(raw.info["sfreq"])
        bip, bip_names, bip_members, bip_pos = build_bipolar(raw, df_elec)
        del raw
        gc.collect()
        try:
            sites = get_stim_events(df_evt)
        except RuntimeError as e:
            print(f"  run {run} skipped: {e}")
            continue

        coord_lut = {}
        if {"name", "x", "y", "z"}.issubset(df_elec.columns):
            for _, r in df_elec.iterrows():
                v = pd.to_numeric(pd.Series([r["x"], r["y"], r["z"]]), errors="coerce").values
                if np.all(np.isfinite(v)):
                    coord_lut[str(r["name"])] = v

        ep = None
        for i_site, (site, onsets) in enumerate(sites.items(), 1):
            c1, c2 = parse_site(site)
            ep, times = make_epochs(bip, sfreq_orig, onsets, cfg["tmin"], cfg["tmax"])
            if ep is None or ep.shape[0] < 3:
                continue
            b0, b1 = np.searchsorted(times, cfg["baseline"])
            baseline_slice = slice(b0, b1)
            p1, p2 = coord_lut.get(c1), coord_lut.get(c2)
            stim_pos = 0.5 * (p1 + p2) if (p1 is not None and p2 is not None) else None

            for j, resp in enumerate(bip_names):
                n_total += 1
                m1, m2 = bip_members[resp]
                if m1 in (c1, c2) or m2 in (c1, c2):      # criterion (1)
                    continue
                ep_ch = ep[:, j, :]
                ok, _ = screen_pair(ep_ch, times, sfreq_orig, cfg, baseline_slice)
                if not ok:
                    continue

                # volume conduction: pairs closer than 10 mm, or of unknown distance,
                # are excluded
                if stim_pos is not None and np.all(np.isfinite(bip_pos[j])):
                    dist = float(np.linalg.norm(bip_pos[j] - stim_pos))
                else:
                    dist = np.nan
                w = volume_weight(dist, cfg["dist_min_mm"], cfg["dist_max_mm"])
                if not np.isfinite(w) or w == 0.0:
                    n_excl_dist += 1
                    continue

                ep_clean = remove_artifact(ep_ch, times, sfreq_orig,
                                           cfg["artifact_ms"], cfg["artifact_pre_ms"])
                db, t_used, _ = ccsr_db(ep_clean, times, sfreq_orig, freqs, cfg)
                if zones is None:
                    zones = tfz_slices(freqs, t_used)
                fracs = significance_fractions(significance_mask(db, t_used, freqs, cfg),
                                               zones)

                rec = dict(session=ses, run=run, stim=site, resp=resp,
                           n_trials=int(ep_ch.shape[0]), dist_mm=dist, weight=w)
                for k, v in fracs.items():
                    rec[k] = v
                    rec[k + "_w"] = v * w
                records.append(rec)
                n_kept += 1

            if i_site % 10 == 0:
                gc.collect()
            print(f"\r  run {run} | site {i_site}/{len(sites)} | pairs kept {n_kept}",
                  end="", flush=True)
        print()
        del bip, ep
        gc.collect()

    summary = dict(subject=subject, n_pairs_evaluated=n_total, n_pairs_kept=n_kept,
                   n_pairs_excluded_distance=n_excl_dist)
    return (pd.DataFrame(records) if records else None), summary


# =============================================================================
# MAIN
# =============================================================================

def run_one(subject, bids_root, out_dir, cfg):
    table, summary = process_subject(subject, bids_root, cfg)
    if table is None:
        print(f"sub-{subject}: no pair retained")
        return 1
    table.to_csv(os.path.join(out_dir, f"sub-{subject}_significance_fractions.csv"),
                 index=False)
    with open(os.path.join(out_dir, f"sub-{subject}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"sub-{subject}: {summary['n_pairs_kept']} pairs kept out of "
          f"{summary['n_pairs_evaluated']} evaluated")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bids-root", required=True, help="local copy of ds004080 v1.2.4")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-ms", type=float, default=CFG["artifact_ms"],
                    help="end of the replaced artefact window (15; 35 for the robustness run)")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--force", action="store_true", help="recompute existing subjects")
    ap.add_argument("--one", help=argparse.SUPPRESS)       # internal: one subject
    args = ap.parse_args()

    cfg = dict(CFG, artifact_ms=args.artifact_ms)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.one:
        sys.exit(run_one(args.one, args.bids_root, args.out_dir, cfg))

    # one fresh interpreter per subject: a run can exceed several GB in memory
    failed = []
    for subject in args.subjects:
        csv = os.path.join(args.out_dir, f"sub-{subject}_significance_fractions.csv")
        if os.path.exists(csv) and not args.force:
            print(f"sub-{subject}: already computed")
            continue
        print(f"sub-{subject}")
        code = subprocess.run([sys.executable, os.path.abspath(__file__),
                               "--bids-root", args.bids_root, "--out-dir", args.out_dir,
                               "--artifact-ms", str(args.artifact_ms), "--one", subject]
                              ).returncode
        if code:
            failed.append(subject)
    print("failed: " + " ".join(failed) if failed else "all subjects done")


if __name__ == "__main__":
    main()
