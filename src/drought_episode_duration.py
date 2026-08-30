"""
A different dimension of "drought"
Run:
    python -m src.drought_episode_duration --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data_loading import load_config
from src.detect_heatwaves import find_episodes

WINDOW = "9m"
MIN_DURATION_DAYS = 1
MIN_EPISODES_FOR_INFERENCE = 8


def get_threshold(thresholds: pd.DataFrame, variable: str, window: str, set_name: str = "primary") -> float:
    row = thresholds[(thresholds["set"] == set_name) & (thresholds["variable"] == variable) & (thresholds["window"] == window)]
    return float(row["threshold_value"].iloc[0])


def compute_persistence(sapei: pd.DataFrame, threshold: float) -> pd.DataFrame:
    sapei = sapei.sort_values("date").reset_index(drop=True)
    sapei["is_drought"] = sapei[f"sapei_{WINDOW}"] < threshold

    episodes = find_episodes(sapei, "is_drought", MIN_DURATION_DAYS)

    sapei["days_into_episode"] = 0
    sapei["episode_total_duration"] = 0
    sapei["episode_id"] = -1
    for i, ep in episodes.iterrows():
        mask = (sapei["date"] >= ep["start_date"]) & (sapei["date"] <= ep["end_date"])
        idx = sapei.index[mask]
        sapei.loc[idx, "days_into_episode"] = np.arange(1, len(idx) + 1)
        sapei.loc[idx, "episode_total_duration"] = ep["duration_days"]
        sapei.loc[idx, "episode_id"] = i

    return sapei


def build_episode_level_table(in_drought: pd.DataFrame, wq_cols: list) -> pd.DataFrame:
    agg = {"date": "count", "episode_total_duration": "first"}
    agg.update({col: "mean" for col in wq_cols})
    episode_level = in_drought.groupby("episode_id").agg(agg).rename(columns={"date": "n_acquisitions"})
    return episode_level.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drought episode persistence vs TSM (with pseudoreplication check)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    sapei = pd.read_csv(processed_dir / "sapei_daily.csv", parse_dates=["date"])
    thresholds = pd.read_csv(processed_dir / "event_thresholds.csv")
    dataset = pd.read_csv(processed_dir / "analysis_dataset.csv", parse_dates=["date"])

    threshold = get_threshold(thresholds, "sapei", WINDOW)
    persistence = compute_persistence(sapei, threshold)

    merged = dataset.merge(
        persistence[["date", "is_drought", "days_into_episode", "episode_total_duration", "episode_id"]],
        on="date", how="left",
    )
    in_drought = merged[merged["is_drought"] == True].copy()  # noqa: E712
    wq_cols = ["tsm", "chl_a", "cdom"]

    n_episodes = in_drought["episode_id"].nunique()
    print(f"Acquisitions in a SAPEI_{WINDOW} drought episode: {len(in_drought)}/{len(merged)}")
    print(f"These cluster inside only {n_episodes} DISTINCT episodes -- that is the true n for any")
    print(f"episode-level question, not {len(in_drought)}.\n")

    acq_records = []
    for predictor in ["days_into_episode", "episode_total_duration"]:
        for y in wq_cols:
            sub = in_drought[[predictor, y]].dropna()
            if len(sub) < 8:
                continue
            r, p = spearmanr(sub[predictor], sub[y])
            acq_records.append({"predictor": predictor, "y": y, "n_acquisitions": len(sub), "spearman_r": r, "p_value": p})
    acq_results = pd.DataFrame(acq_records)
    acq_results.to_csv(tables_dir / "drought_episode_duration_acquisition_level.csv", index=False)

    episode_level = build_episode_level_table(in_drought, wq_cols)
    episode_level.to_csv(tables_dir / "drought_episode_duration_episode_level.csv", index=False)

    lines = ["DROUGHT EPISODE PERSISTENCE vs WQ", "=" * 70]
    lines.append(
        f"\n{len(in_drought)} acquisitions fall inside a drought episode, but these cluster inside "
        f"only {n_episodes} distinct episodes."
    )
    lines.append("\nAcquisition-level correlations (PSEUDOREPLICATED -- shown for transparency, not as a result):")
    for _, row in acq_results.iterrows():
        lines.append(f"  {row['predictor']:<25} vs {row['y']:<8} r={row['spearman_r']:+.3f}  p={row['p_value']:.4f}  (n={row['n_acquisitions']}, but only {n_episodes} independent episodes)")

    lines.append(f"\nEpisode-level table (n={n_episodes}, the correct unit of analysis):")
    lines.append(episode_level.to_string(index=False))

    if n_episodes < MIN_EPISODES_FOR_INFERENCE:
        lines.append(
            f"\nCONCLUSION: only {n_episodes} independent drought episodes are captured by the "
            f"Sentinel-2 archive -- far too few for any correlation test between episode duration "
            f"and WQ to have meaningful statistical power. No p-value is reported at the episode "
            f"level. The honest conclusion is that this record cannot currently test whether "
            f"drought PERSISTENCE (as opposed to instantaneous severity, already tested in "
            f"src.correlation_analysis) matters -- not that it doesn't."
        )
    summary = "\n".join(lines)
    (tables_dir / "drought_episode_duration_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
