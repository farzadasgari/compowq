"""
Compute the daily-scale Standardized Compound Drought-Heat Index (SCDHI).

Run:
    python -m src.compute_scdhi --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from src.data_loading import load_config
from src.copulas import FAMILIES, compare_families


def to_uniform_empirical(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, method="average")
    n = len(x)
    return ranks / (n + 1)


def compute_scdhi_for_window(sapei: pd.Series, sti: pd.Series, primary_family: str):
    mask = sapei.notna() & sti.notna()
    if mask.sum() < 30:
        return pd.Series(index=sapei.index, dtype=float), []

    u_all = norm.cdf(sapei)
    v_all = norm.cdf(sti)
    u = np.clip(u_all[mask.values], 1e-6, 1 - 1e-6)
    v = np.clip(v_all[mask.values], 1e-6, 1 - 1e-6)

    comparison = compare_families(u, v)
    chosen = next(
        (r for r in comparison if r["family"] == primary_family), None)
    if chosen is None or chosen["params"] is None:
        raise ValueError(
            f"Copula family '{primary_family}' failed to fit; check scdhi_copula_comparison.csv")

    c = FAMILIES[primary_family]["cdf"](u, v, chosen["params"])
    p = np.clip(u - c, 1e-10, 1 - 1e-10)
    q = to_uniform_empirical(p)
    scdhi_values = norm.ppf(q)

    scdhi = pd.Series(index=sapei.index, dtype=float)
    scdhi[mask] = scdhi_values
    return scdhi, comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute daily SCDHI against all SAPEI windows")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    sti_path = processed_dir / "sti_daily.csv"
    sapei_path = processed_dir / "sapei_daily.csv"
    if not sti_path.exists():
        raise SystemExit(
            f"ERROR: {sti_path} not found. Run `python -m src.compute_sti --config {args.config}` first.")
    if not sapei_path.exists():
        raise SystemExit(
            f"ERROR: {sapei_path} not found. Run `python -m src.compute_sapei --config {args.config}` first.")

    primary_family = cfg.get("indices", {}).get(
        "scdhi", {}).get("copula_family", "frank")
    if primary_family not in FAMILIES:
        raise SystemExit(
            f"ERROR: indices.scdhi.copula_family = '{primary_family}' not in {list(FAMILIES)}")

    sti = pd.read_csv(sti_path, parse_dates=["date"])[["date", "sti"]]
    sapei = pd.read_csv(sapei_path, parse_dates=["date"])

    merged = sapei.merge(sti, on="date", how="inner")
    sapei_cols = sorted(c for c in sapei.columns if c.startswith("sapei_"))

    out = merged[["date"]].copy()
    comparison_rows = []
    print(
        f"SCDHI computed {merged['date'].min().date()} -> {merged['date'].max().date()}")
    print(
        f"Primary copula family: {primary_family} (configs/config.yaml: indices.scdhi.copula_family)\n")

    for col in sapei_cols:
        window = col.replace("sapei_", "")
        scdhi, comparison = compute_scdhi_for_window(
            merged[col], merged["sti"], primary_family)
        out[f"scdhi_{window}"] = scdhi

        print(
            f"  window {window} -- AIC comparison (lower is better), best first:")
        for rank_i, r in enumerate(comparison, start=1):
            marker = " <- used" if r["family"] == primary_family else ""
            if r["params"] is None:
                print(
                    f"    {rank_i}. {r['family']:<9} FAILED: {r.get('error')}")
            else:
                print(
                    f"    {rank_i}. {r['family']:<9} AIC={r['aic']:9.2f}  loglik={r['loglik']:9.2f}  params={r['params']}{marker}")
            comparison_rows.append({"window": window, "rank": rank_i, **r})

        n_valid = scdhi.notna().sum()
        if n_valid:
            print(
                f"     -> scdhi_{window} valid: {n_valid}/{len(out)}   range: {scdhi.min():.3f} to {scdhi.max():.3f}\n")
        else:
            print(
                f"     -> scdhi_{window} skipped (insufficient paired data)\n")

    out_path = processed_dir / "scdhi_daily.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    comparison_path = processed_dir / "scdhi_copula_comparison.csv"
    pd.DataFrame(comparison_rows).drop(
        columns=["error"], errors="ignore").to_csv(comparison_path, index=False)
    print(f"Wrote {comparison_path}")

    wq_path = processed_dir / "wq_clean.csv"
    if wq_path.exists():
        wq = pd.read_csv(wq_path, parse_dates=["date"])
        matched = wq.merge(out, on="date", how="left")
        print(
            f"\nSanity check -- exact-date match against {len(wq)} Sentinel-2 acquisitions:")
        for col in out.columns:
            if col == "date":
                continue
            n_matched = matched[col].notna().sum()
            print(f"  {col:<10} {n_matched}/{len(wq)} acquisition dates matched")


if __name__ == "__main__":
    main()
