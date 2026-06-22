"""Stronger cheap difficulty baselines (Exp8).

The §4.2/§4.3 P1 baseline is eight classical stats. A reviewer rightly asks: is
the SAE null just because P1 is a weak baseline? So we add the strongest cheap,
label-free predictors of forecast difficulty and make the SAE beat *those*:

  - STL trend strength    : 1 - Var(R)/Var(T+R)   (Wang/Hyndman tsfeatures)
  - STL seasonal strength : 1 - Var(R)/Var(S+R)
  - Hurst exponent        : R/S long-memory estimate (0.5=random, >0.5 trending,
                            <0.5 mean-reverting) -- persistence is a strong proxy
                            for how forecastable the continuation is.
  - spectral entropy      : already in P1 (compute_input_stats); kept there.
  - forecast-interval width: the model's OWN predictive band (band_width_90 from
                            eval/extract_uncertainty.py). This is the natural
                            difficulty signal the SAE must beat; merged from the
                            uncertainty parquet by `interval_width_feature`.

These are pure functions of the context window (Hurst/STL) or of the saved
forecast (interval width) -- no TSFM forward pass -- so they are cheap and
testable offline. Imported by probing/probe.py behind --extended_baselines.
"""
import numpy as np


def stl_strength(x, period):
    """(trend_strength, seasonal_strength) in [0,1] via STL. Falls back to
    (0,0) when the window is too short for the requested period."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if period < 2 or n < 2 * period + 1:
        return 0.0, 0.0
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(x, period=period, robust=True).fit()
        remainder = res.resid
        var_r = np.var(remainder)
        denom_t = np.var(res.trend + remainder)
        denom_s = np.var(res.seasonal + remainder)
        ts = 0.0 if denom_t <= 0 else max(0.0, 1.0 - var_r / denom_t)
        ss = 0.0 if denom_s <= 0 else max(0.0, 1.0 - var_r / denom_s)
        return float(ts), float(ss)
    except Exception:
        return 0.0, 0.0


def hurst_exponent(x):
    """Rescaled-range (R/S) Hurst estimate. ~0.5 = uncorrelated, >0.5 =
    persistent/trending, <0.5 = mean-reverting."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20:
        return 0.5
    lags = np.unique(np.floor(np.logspace(np.log10(8), np.log10(N // 2), 12)).astype(int))
    lags = lags[lags >= 4]
    pts = []
    for lag in lags:
        n_chunks = N // lag
        if n_chunks < 1:
            continue
        rs_vals = []
        for c in range(n_chunks):
            seg = x[c * lag:(c + 1) * lag]
            z = seg - seg.mean()
            Z = np.cumsum(z)
            R = Z.max() - Z.min()
            S = seg.std()
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            pts.append((lag, np.mean(rs_vals)))
    if len(pts) < 2:
        return 0.5
    lg, rs = zip(*pts)
    H = np.polyfit(np.log(lg), np.log(rs), 1)[0]
    return float(np.clip(H, 0.0, 1.5))


def compute_extended_baselines(df_meta, series, context_length=512, season_length=24):
    """Per-window [trend_strength, seasonal_strength, hurst] over the context
    window. `series` is the full channel array the windows index into."""
    series = np.asarray(series, dtype=float)
    out = []
    for _, row in df_meta.iterrows():
        s = int(row["start_ts"])
        x = series[s:s + context_length]
        ts, ss = stl_strength(x, season_length)
        out.append([ts, ss, hurst_exponent(x)])
    return np.array(out)


def interval_width_feature(df_meta, uncertainty_parquet, width_col="band_width_90"):
    """Merge the model's own forecast-interval width (the natural difficulty
    signal) onto the windows by start_ts. Returns an (n,1) column; NaN where a
    window has no saved forecast (e.g. uncertainty extraction was test-only)."""
    import pandas as pd
    unc = pd.read_parquet(uncertainty_parquet)[["start_ts", width_col]]
    merged = df_meta[["start_ts"]].merge(unc, on="start_ts", how="left")
    return merged[[width_col]].to_numpy()


def _selftest():
    rng = np.random.default_rng(0)
    n = 512
    t = np.arange(n)
    white = rng.standard_normal(n)
    rw = np.cumsum(rng.standard_normal(n))                       # random walk: persistent
    trend = 0.05 * t + 0.3 * rng.standard_normal(n)             # strong trend
    seasonal = 3 * np.sin(2 * np.pi * t / 24) + 0.3 * rng.standard_normal(n)  # strong daily

    h_white, h_rw = hurst_exponent(white), hurst_exponent(rw)
    tr_trend, _ = stl_strength(trend, 24)
    _, se_seas = stl_strength(seasonal, 24)
    tr_white, se_white = stl_strength(white, 24)

    print(f"Hurst:   white={h_white:.2f} (~0.5)   random-walk={h_rw:.2f} (>0.7)")
    print(f"STL trend strength:    trended={tr_trend:.2f} (>0.6)   white={tr_white:.2f} (<0.3)")
    print(f"STL seasonal strength: seasonal={se_seas:.2f} (>0.6)   white={se_white:.2f} (<0.3)")
    ok = (h_white < 0.65 and h_rw > 0.7 and tr_trend > 0.6
          and se_seas > 0.6 and tr_white < 0.4 and se_white < 0.4)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
