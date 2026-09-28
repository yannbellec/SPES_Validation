#!/usr/bin/env python3
"""Stage 3 - figure panels of the paper (one PNG per panel, in --fig-dir).

Figure 1A-C re-read the raw data of the illustrated subject; all other panels use the
CSVs of pipeline.py and the tables written by analysis.py (run analysis.py first).

Usage:
    python figures.py --bids-root data/ds004080 --out-dir outputs/art15 \
                      --results-dir results --fig-dir figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import analysis as A
import pipeline as P

FIG1_SUBJECT = "ccepAgeUMCU58"
FIG2_SUBJECTS = ("ccepAgeUMCU21", "ccepAgeUMCU58")
FIG4_SUBJECTS = ("ccepAgeUMCU58", "ccepAgeUMCU21", "ccepAgeUMCU46")
TOP_MARKS = (0.1, 1.0, 5.0)
DPI = 220


def save(fig, fig_dir, name):
    fig.savefig(os.path.join(fig_dir, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURE 1 - pipeline illustrated on one subject
# =============================================================================

def artifact_latency_ms(trace_uv, times, baseline_slice, search_ms=120.0,
                        n_mad=6.0, floor_uv=25.0):
    """Robust artefact end (ms), used only to choose the illustrated pair: last sample
    within search_ms whose deviation exceeds max(6 x MAD-based SD, 25 uV)."""
    base = trace_uv[baseline_slice]
    med = np.median(base)
    mad = np.median(np.abs(base - med))
    thr = max(n_mad * (1.4826 * mad if mad > 0 else np.std(base)), floor_uv)
    fs = 1.0 / np.median(np.diff(times))
    i0 = int(np.argmin(np.abs(times)))
    above = np.abs(trace_uv[i0:min(i0 + int(round(search_ms * 1e-3 * fs)), len(times))]
                   - med) > thr
    return float(np.where(above)[0][-1] + 1) / fs * 1000.0 if above.any() else 0.0


def illustrated_pair(subject, bids_root, cfg=P.CFG, top_n=5):
    """Among the pairs passing criteria (1), (3), distance and artefact latency, the one
    with the largest N2 deflection (50-500 ms) of the mean evoked potential."""
    cands = []
    for ses, run in P.find_runs(bids_root, subject, cfg["task"]):
        raw, df_elec, _, df_evt = P.load_run(bids_root, subject, ses, run, cfg["task"])
        sfreq = float(raw.info["sfreq"])
        bip, names, members, pos = P.build_bipolar(raw, df_elec)
        del raw
        lut = {}
        for _, r in df_elec.iterrows():
            v = pd.to_numeric(pd.Series([r["x"], r["y"], r["z"]]), errors="coerce").values
            if np.all(np.isfinite(v)):
                lut[str(r["name"])] = v
        for site, onsets in P.get_stim_events(df_evt).items():
            c1, c2 = P.parse_site(site)
            ep, times = P.make_epochs(bip, sfreq, onsets, cfg["tmin"], cfg["tmax"])
            if ep is None or ep.shape[0] < 3:
                continue
            b0, b1 = np.searchsorted(times, cfg["baseline"])
            spos = 0.5 * (lut[c1] + lut[c2]) if (c1 in lut and c2 in lut) else None
            for j, resp in enumerate(names):
                m1, m2 = members[resp]
                if m1 in (c1, c2) or m2 in (c1, c2):
                    continue
                ep_ch = ep[:, j, :]
                if np.mean(np.std(ep_ch, axis=-1) * 1e6 > cfg["std_threshold_uv"]) \
                        > cfg["bad_trial_fraction"]:
                    continue
                dist = (float(np.linalg.norm(pos[j] - spos))
                        if spos is not None and np.all(np.isfinite(pos[j])) else np.nan)
                w = P.volume_weight(dist, cfg["dist_min_mm"], cfg["dist_max_mm"])
                if not np.isfinite(w) or w == 0.0:
                    continue
                ep_clean = P.remove_artifact(ep_ch, times, sfreq,
                                             cfg["artifact_ms"], cfg["artifact_pre_ms"])
                ccep = ep_clean.mean(axis=0) * 1e6
                ccep = ccep - np.mean(ccep[b0:b1])
                lat = artifact_latency_ms(ccep, times, slice(b0, b1))
                if np.isfinite(lat) and lat > cfg["max_artifact_latency_ms"]:
                    continue
                n2 = (times >= 0.050) & (times <= 0.500)
                score = abs(float(ccep[n2][int(np.argmax(np.abs(ccep[n2])))]))
                if not np.isfinite(score):
                    continue
                cands.append(dict(score=score, site=site, resp=resp, times=times,
                                  sfreq=sfreq, ccep=ccep, ep_clean=ep_clean,
                                  n_trials=int(ep_ch.shape[0])))
                cands.sort(key=lambda d: -d["score"])
                del cands[top_n:]
    return cands[0]


def figure1(bids_root, out_dir, fig_dir, cfg=P.CFG):
    c = illustrated_pair(FIG1_SUBJECT, bids_root, cfg)
    freqs = P.make_freqs(cfg["fmin"], cfg["fmax"], cfg["voices_per_octave"])
    db, t, _ = P.ccsr_db(c["ep_clean"], c["times"], c["sfreq"], freqs, cfg)
    mask = P.significance_mask(db, t, freqs, cfg)
    t_ms = t * 1000.0
    sel = (t_ms >= -300) & (t_ms <= 600)

    # share of significant pixels in the test baseline and in the whole-response zone
    b0, b1 = np.searchsorted(t, cfg["baseline"])
    fsel, tsel = P.tfz_slices(freqs, t)["WR"]
    print(f"Figure 1 pair {c['site']} -> {c['resp']}: significant pixels "
          f"{100 * mask[:, b0:b1].mean():.0f} % in the baseline, "
          f"{100 * mask[np.ix_(fsel, tsel)].mean():.0f} % in the response zone")

    # A - mean evoked potential
    tt = c["times"] * 1000.0
    s = (tt >= -300) & (tt <= 600)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axvspan(-cfg["artifact_pre_ms"], cfg["artifact_ms"], color="0.87",
               label=f"replaced artefact window (-{cfg['artifact_pre_ms']:g} to "
                     f"+{cfg['artifact_ms']:g} ms)")
    ax.axvspan(15, 50, color="#ffe6a8", alpha=.55, label="N1 zone (15-50 ms)")
    ax.axvspan(50, 500, color="#d6ecff", alpha=.45, label="N2 zone (50-500 ms)")
    ax.plot(tt[s], c["ccep"][s], color="#1f4fd8", lw=2,
            label=f"mean CCEP (n = {c['n_trials']} trials)")
    ax.axvline(0, color="black", ls="--", lw=1.2)
    ax.set(xlim=(-300, 600), xlabel="Time (ms)", ylabel="Amplitude (µV)")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, fig_dir, "fig1A_ccep.png")

    # B - mean spectral response, evoked potential overlaid in arbitrary units
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    pcm = ax.pcolormesh(t_ms[sel], freqs, db[:, sel], cmap="jet", shading="gouraud",
                        vmin=-10, vmax=10)
    ax.set_yscale("log")
    ccep = np.interp(t, c["times"], c["ccep"])
    lo, hi = np.percentile(ccep[sel], [1, 99])
    nrm = np.clip((ccep - lo) / ((hi if hi > lo else lo + 1.0) - lo), 0, 1)
    l0, l1 = np.log10(freqs[0]), np.log10(freqs[-1])
    ax.plot(t_ms[sel], (10 ** (l0 + .10 * (l1 - l0) + nrm * .80 * (l1 - l0)))[sel],
            color="white", lw=2, label="mean CCEP (rescaled)")
    ax.set(xlim=(-300, 600), ylim=(freqs[0], freqs[-1]), xlabel="Time (ms)",
           ylabel="Frequency (Hz)")
    ax.set_yticks([4, 9, 22, 50, 116, 250], ["4", "9", "22", "50", "116", "250"])
    ax.legend(fontsize=8.5, loc="upper right")
    fig.colorbar(pcm, ax=ax, pad=.02, fraction=.05, label="Power (dB re baseline)")
    save(fig, fig_dir, "fig1B_ccsr.png")

    # C - significant pixels after Benjamini-Hochberg, with the 11 zones (whole epoch
    # start shown: the wavelet edge effect appears at the left border)
    full = (t_ms >= -500) & (t_ms <= 600)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.pcolormesh(t_ms[full], freqs, mask[:, full], cmap=ListedColormap(["gray", "white"]),
                  shading="nearest")
    ax.set_yscale("log")
    ax.set(xlim=(-500, 600), ylim=(freqs[0], freqs[-1]), xlabel="Time (ms)",
           ylabel="Frequency (Hz)")
    ax.set_yticks([4, 9, 22, 50, 116, 250], ["4", "9", "22", "50", "116", "250"])
    for f in (8, 13, 30, 50):
        ax.axhline(f, color="black", lw=1.2)
    for x in (15, 50, 500):
        ax.axvline(x, color="black", lw=1.2)
    for lbl, fpos in (("T", 5.5), ("A", 10), ("B", 20), ("Gl", 39), ("Gh", 110)):
        ax.text(-40, fpos, lbl, fontsize=13, fontweight="bold", va="center", ha="right")
    ax.add_patch(patches.Rectangle((15, 4), 485, 246, lw=2.5, edgecolor="red",
                                   facecolor="none"))
    ax.text(515, 6, "WR", color="red", fontsize=14, fontweight="bold", va="top")
    save(fig, fig_dir, "fig1C_significant_pixels.png")

    # D - GhN2 significance fractions of all pairs
    matrix_panel(out_dir, bids_root, FIG1_SUBJECT, fig_dir, "fig1D_ghn2_matrix.png")


# =============================================================================
# SUBJECT MATRICES AND CURVES (Figures 1D, 2B-C, 4)
# =============================================================================

def build_matrix(df, tfz="GhN2"):
    """Stimulated x response matrix of the weighted fractions (NaN = excluded pair)."""
    stims = sorted(df["stim"].astype(str).unique())
    resps = sorted(df["resp"].astype(str).unique())
    si = {s: i for i, s in enumerate(stims)}
    ri = {r: i for i, r in enumerate(resps)}
    M = np.full((len(stims), len(resps)), np.nan)
    for s, r, v in zip(df["stim"].astype(str), df["resp"].astype(str), df[tfz + "_w"]):
        M[si[s], ri[r]] = v
    return M, stims, resps


def label_flags(names, labels):
    return np.array([any(c in labels for c in P.parse_site(str(n))) for n in names])


def mark_labels(ax, M, stim_soz, resp_soz, ms=3):
    """Red squares outside the axes for the channels with a SOZ contact."""
    n_s, n_r = M.shape
    for j in np.where(resp_soz)[0]:
        ax.plot(j, n_s + 0.012 * n_s, "s", color="red", ms=ms, clip_on=False, zorder=5)
    for i in np.where(stim_soz)[0]:
        ax.plot(-0.012 * n_r, i, "s", color="red", ms=ms, clip_on=False, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])


def subject_matrix(out_dir, bids_root, subject):
    df = pd.read_csv(os.path.join(out_dir, f"sub-{subject}_significance_fractions.csv"))
    soz = P.labelled_contacts(P.electrodes_table(bids_root, subject), "soz")
    M, stims, resps = build_matrix(df)
    return M, label_flags(stims, soz), label_flags(resps, soz)


def matrix_panel(out_dir, bids_root, subject, fig_dir, name):
    M, stim_soz, resp_soz = subject_matrix(out_dir, bids_root, subject)
    cm = plt.get_cmap("jet").copy()
    cm.set_bad(color="0.45")
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    mark_labels(ax, M, stim_soz, resp_soz)
    ax.set(xlabel="Response channels", ylabel="Stimulated channels")
    fig.colorbar(im, ax=ax, pad=0.03, fraction=0.05, label="GhN2 significance fraction")
    save(fig, fig_dir, name)


def top_pairs_panel(out_dir, bids_root, subject, fig_dir, name):
    """The same matrix restricted to the top 0.1, 1 and 5 % of fractions."""
    M, stim_soz, resp_soz = subject_matrix(out_dir, bids_root, subject)
    finite = M[np.isfinite(M)]
    cm = plt.get_cmap("jet").copy()
    cm.set_bad(color="black")
    fig, axes = plt.subplots(1, len(TOP_MARKS), figsize=(4.6 * len(TOP_MARKS), 4.4))
    for ax, xp in zip(axes, TOP_MARKS):
        k = max(int(np.ceil(finite.size * xp / 100.0)), 1)
        thr = np.sort(finite)[::-1][k - 1]
        ax.imshow(np.ma.masked_invalid(np.where(M >= thr, M, np.nan)), cmap=cm, vmin=0,
                  vmax=1, aspect="auto", interpolation="nearest")
        mark_labels(ax, M, stim_soz, resp_soz, ms=2.5)
        ax.set_title(f"Top {xp:g} %  ({k} pairs)")
        ax.set_xlabel("Response channels")
    axes[0].set_ylabel("Stimulated channels")
    fig.suptitle(f"sub-{subject}")
    save(fig, fig_dir, name)


def curves_panel(subs, subject, fig_dir, name, tfz="GhN2"):
    """Specificity and sensitivity of one subject, with the 95th percentile of its own
    surrogates (the reference used in analysis.per_subject_table)."""
    s = next(r for r in subs if r["subject"] == subject)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, key, title in zip(axes, ("spec", "sens"), ("Specificity", "Sensitivity")):
        curve, ref = s["true"][tfz][key], s["surr95"][tfz][key]
        ax.plot(A.X_PERCENT, curve, color="black", lw=2, label="observed")
        ax.plot(A.X_PERCENT, ref, color="#d62728", lw=1.4, ls=":",
                label="95th percentile of surrogates")
        for xp in TOP_MARKS:
            i = A.x_index(xp)
            ax.plot(xp, curve[i], "*", color="#1f4fd8", ms=13)
            ax.annotate(f"{curve[i]:.0f}%", (xp, curve[i]), textcoords="offset points",
                        xytext=(6, 6), fontsize=9)
        ax.set(xlim=(0, 10), ylim=(0, 100), xlabel="Top X %", ylabel="Proportion (%)",
               title=title)
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"sub-{subject}")
    save(fig, fig_dir, name)


# =============================================================================
# GROUP FIGURES (2A, 3, 5)
# =============================================================================

def group_curves(subs, fig_dir, name):
    """Figure 2A: specificity of every subject (grey), group median (black), median and
    95th percentile of the surrogate medians (red), significant X values (blue)."""
    res = A.group_analysis(subs)
    tfzs = [t for t in A.TFZ_ORDER if t in res]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for ax, tfz in zip(axes.ravel(), tfzs):
        r = res[tfz]
        for row in r["spec_indiv"]:
            ax.plot(r["x"], row, color="0.75", lw=.7, zorder=1)
        ax.plot(r["x"], r["spec_rand"], color="#d62728", lw=2, zorder=3)
        ax.plot(r["x"], r["spec_true"], color="black", lw=2, zorder=4)
        ax.plot(r["x"], r["spec_95"], color="#d62728", lw=1, ls=":", zorder=3)
        sig = r["spec_sig"]
        if sig.any():
            ax.plot(r["x"][sig], np.full(sig.sum(), 98), "s", color="#1f4fd8", ms=2.5)
        ax.set_title(f"{tfz}  ({100 * sig.mean():.0f}% of X)", fontsize=11)
        ax.set(xlim=(0, 10), ylim=(0, 100))
    for ax in axes.ravel()[len(tfzs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Top X %")
    for ax in axes[:, 0]:
        ax.set_ylabel("Specificity (%)")
    fig.tight_layout()
    save(fig, fig_dir, name)


def targets_panel(results_dir, fig_dir, name):
    """Figure 3: subject-level margin and 95 % CI for the two clinical targets."""
    d = pd.read_csv(os.path.join(results_dir, "subject_level_bootstrap.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    for off, (target, col) in zip((-0.18, 0.18), (("hypothesised SOZ", "#4878a8"),
                                                  ("resected tissue", "#5a9367"))):
        t = d[d["target"] == target].set_index("zone").loc[A.TFZ_ORDER]
        x = np.arange(len(t)) + off
        ax.errorbar(x, t["margin"], yerr=[t["margin"] - t["ci_low"], t["ci_high"] - t["margin"]],
                    fmt="o", capsize=3, color=col, label=target, ms=6, lw=1.5)
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_xticks(np.arange(len(A.TFZ_ORDER)), A.TFZ_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Margin over the median of surrogates (points, 95 % CI)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, fig_dir, name)


def noise_panel(results_dir, fig_dir, name):
    """Figure 5A: per-subject margin over X <= 2 % for three rankings of the pairs."""
    d = pd.read_csv(os.path.join(results_dir, "noise_competitors.csv"))
    data = [d["margin_ghn2"], d["margin_noise"], d["margin_residual"]]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bp = ax.boxplot(data, tick_labels=["GhN2 fraction", "Baseline noise only",
                                       "GhN2 residualised\non noise"],
                    showmeans=True, widths=0.55, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#4878a8", "#c0554f", "#5a9367"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    rng = np.random.default_rng(0)                  # horizontal jitter of the points
    for i, v in enumerate(data, 1):
        ax.scatter(rng.normal(i, 0.05, len(v)), v, s=14, color="0.25", alpha=0.6, zorder=3)
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_ylabel("Margin over chance, X ≤ 2 % (specificity points)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, fig_dir, name)


def anatomy_panel(results_dir, fig_dir, name):
    """Figure 5B: observed margin difference against the label permutations."""
    obs = pd.read_csv(os.path.join(results_dir, "anatomy_test.csv")).iloc[0]
    perm = pd.read_csv(os.path.join(results_dir, "anatomy_permutations.csv"))["difference"]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.hist(perm, bins=40, color="#b8c4d0", edgecolor="white")
    ax.axvline(obs["difference"], color="#d62728", lw=2.5,
               label=f"observed ({obs['difference']:+.1f}, p = {obs['p']:.3f})")
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.set(xlabel="Margin difference, temporal − extratemporal (points)",
           ylabel="Label permutations")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, fig_dir, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--out-dir", required=True, help="CSVs of the 15 ms run")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--skip-raw", action="store_true", help="skip Figure 1A-C (raw data)")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    subs = A.subject_results(args.out_dir, args.bids_root, "soz")

    if args.skip_raw:
        matrix_panel(args.out_dir, args.bids_root, FIG1_SUBJECT, args.fig_dir,
                     "fig1D_ghn2_matrix.png")
    else:
        figure1(args.bids_root, args.out_dir, args.fig_dir)
    group_curves(subs, args.fig_dir, "fig2A_group_specificity.png")
    for letter, sub in zip("BC", FIG2_SUBJECTS):
        top_pairs_panel(args.out_dir, args.bids_root, sub, args.fig_dir,
                        f"fig2{letter}_top_pairs_{sub}.png")
    targets_panel(args.results_dir, args.fig_dir, "fig3_two_targets.png")
    for sub in FIG4_SUBJECTS:
        matrix_panel(args.out_dir, args.bids_root, sub, args.fig_dir, f"fig4_matrix_{sub}.png")
        curves_panel(subs, sub, args.fig_dir, f"fig4_curves_{sub}.png")
    noise_panel(args.results_dir, args.fig_dir, "fig5A_noise.png")
    anatomy_panel(args.results_dir, args.fig_dir, "fig5B_anatomy.png")
    print(f"figures written to {args.fig_dir}")


if __name__ == "__main__":
    main()
