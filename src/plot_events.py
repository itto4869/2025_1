"""TensorBoard events (*.tfevents*) を読み取り学習曲線をプロットするユーティリティ.

主な機能:
- ディレクトリ以下を再帰的に探索し event ファイルを集める
- 指定 (または自動検出) の scalar tag を抽出
- pandas DataFrame に統合 (run 別 / tag 別)
- 移動平均 / 指数移動平均によるスムージング
- 同一 tag の複数 run を平均 + 標準偏差バンドでプロット

Usage (例):

python -m src.plot_events \
    --log-root tensorboard_logs/atari/adam \
    --tags rollout/ep_rew_mean eval/atari_mean_reward \
    --ema 0.9 \
    --output figures/atari_rewards.png

引数:
    --log-root: イベントファイルを含むルート (複数指定可)
    --tags: 抽出したいタグ (未指定なら各 run で上位の代表的タグを自動選択)
    --recursive: ディレクトリを再帰走査 (デフォルト True)
    --ema / --window: スムージング方法 (EMA 係数 or 移動平均 window)
    --output: 画像保存パス (未指定なら表示のみ)
    --xaxis: step | wall_time | index (デフォルト step)
    --no-legend: 凡例非表示

必要ライブラリ: tensorboard (SB3 依存), pandas, matplotlib
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence, cast

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sklearn.metrics import auc as sk_auc  # type: ignore

SCALAR_SIZE_GUIDANCE = {"scalars": 0}
DEFAULT_CANDIDATE_TAGS = [
    # SB3 共通
    "rollout/ep_rew_mean",
    "eval/atari_mean_reward",
    "eval/mujoco_mean_reward",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    # 時間計測
    "time/training_wall_clock_sec",
]


def find_event_files(root: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/events.out.tfevents.*" if recursive else "events.out.tfevents.*"
    return sorted(root.glob(pattern))


def load_scalars_from_event(file_path: Path, tags: Sequence[str] | None, group: str | None = None) -> pd.DataFrame:
    acc = EventAccumulator(str(file_path), size_guidance=SCALAR_SIZE_GUIDANCE)
    try:
        acc.Reload()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to load {file_path}: {e}")
        return pd.DataFrame()

    tags_dict = acc.Tags()
    raw_scalars = tags_dict["scalars"] if "scalars" in tags_dict else []
    if not isinstance(raw_scalars, (list, tuple, set)):
        raw_scalars = []  # fallback safety
    available = {str(t) for t in raw_scalars}
    selected: Sequence[str]
    if tags is None or len(tags) == 0:
        # 自動選択: 代表タグ候補と交差
        selected = [t for t in DEFAULT_CANDIDATE_TAGS if t in available]
        if not selected:  # fallback 全部
            selected = sorted(available)
    else:
        selected = [t for t in tags if t in available]

    rows = []
    run_name = file_path.parent.name
    for tag in selected:
        for e in acc.Scalars(tag):
            rows.append(
                {
                    "run": run_name,
                    "group": group if group is not None else run_name,
                    "tag": tag,
                    "step": e.step,
                    "wall_time": e.wall_time,
                    "value": e.value,
                }
            )
    return pd.DataFrame(rows)


def concat_event_scalars(event_files: Iterable[tuple[Path, str | None]], tags: Sequence[str] | None) -> pd.DataFrame:
    dfs = [load_scalars_from_event(p, tags, group) for p, group in event_files]
    if not dfs:
        return pd.DataFrame({c: [] for c in ["run", "group", "tag", "step", "wall_time", "value"]})
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        return pd.DataFrame({c: [] for c in ["run", "group", "tag", "step", "wall_time", "value"]})
    return pd.concat(non_empty, ignore_index=True)


def apply_smoothing(df: pd.DataFrame, ema: float | None, window: int | None) -> pd.DataFrame:
    if ema is None and (window is None or window <= 1):
        return df

    def _smooth(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("step").copy()
        if ema is not None:
            # 指数移動平均
            g["value_smooth"] = g["value"].ewm(alpha=1 - ema).mean()
        if window is not None and window > 1:
            g["value_ma"] = g["value"].rolling(window=window, min_periods=1).mean()
        return g

    return df.groupby(["run", "tag"], group_keys=False).apply(_smooth)


def plot_tags(
    df: pd.DataFrame,
    xaxis: str = "step",
    use_smooth: bool = True,
    aggregate_runs: bool = True,
    force_aggregate: bool = False,
    legend: bool = True,
    per_group: bool = False,
):
    if df.empty:
        print("[WARN] No data to plot.")
        return

    tags = sorted(cast(Sequence[str], df["tag"].astype(str).unique().tolist()))
    n_cols = min(2, len(tags))
    n_rows = math.ceil(len(tags) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, tag in enumerate(tags):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = df[df.tag == tag]
        value_col = "value_smooth" if use_smooth and "value_smooth" in sub.columns else (
            "value_ma" if use_smooth and "value_ma" in sub.columns else "value"
        )
        if per_group and "group" in sub.columns:
            for group_name, group_df in sub.groupby("group"):
                run_unique_count = len({str(r) for r in group_df["run"]})
                use_agg = (aggregate_runs and run_unique_count > 1) or force_aggregate
                if use_agg:
                    pivot: list[pd.DataFrame] = []
                    for run, g in group_df.groupby("run"):
                        g_df = cast(pd.DataFrame, g)
                        if xaxis in g_df.columns:
                            g_sorted = g_df.sort_values(xaxis)
                        else:
                            g_sorted = g_df.reset_index(drop=True).assign(**{xaxis: range(len(g_df))})
                        g2 = g_sorted[[xaxis, value_col]].set_index(xaxis)
                        pivot.append(g2.rename(columns={value_col: str(run)}))
                    if pivot:
                        merged = pd.concat(pivot, axis=1)
                        merged["mean"] = merged.mean(axis=1, numeric_only=True)
                        merged["std"] = merged.std(axis=1, numeric_only=True)
                        ax.plot(merged.index, merged["mean"], label=f"{group_name} (mean)")
                        ax.fill_between(
                            merged.index,
                            merged["mean"] - merged["std"],
                            merged["mean"] + merged["std"],
                            alpha=0.2,
                        )
                else:
                    for run, g in group_df.groupby("run"):
                        g_df = cast(pd.DataFrame, g)
                        if xaxis in g_df.columns:
                            g_sorted = g_df.sort_values(xaxis)
                        else:
                            g_sorted = g_df.reset_index(drop=True).assign(**{xaxis: range(len(g_df))})
                        ax.plot(g_sorted[xaxis], g_sorted[value_col], label=f"{group_name}:{run}")
        else:
            run_unique_count = len({str(r) for r in sub["run"]})
            use_agg = (aggregate_runs and run_unique_count > 1) or force_aggregate
            if use_agg:
                pivot = []
                for run, g in sub.groupby("run"):
                    g_df = cast(pd.DataFrame, g)
                    if xaxis in g_df.columns:
                        g_sorted = g_df.sort_values(xaxis)
                    else:
                        g_sorted = g_df.reset_index(drop=True).assign(**{xaxis: range(len(g_df))})
                    g2 = g_sorted[[xaxis, value_col]].set_index(xaxis)
                    pivot.append(g2.rename(columns={value_col: str(run)}))
                if pivot:
                    merged = pd.concat(pivot, axis=1)
                    merged["mean"] = merged.mean(axis=1, numeric_only=True)
                    merged["std"] = merged.std(axis=1, numeric_only=True)
                    ax.plot(merged.index, merged["mean"], label=f"{tag} (mean)")
                    ax.fill_between(
                        merged.index,
                        merged["mean"] - merged["std"],
                        merged["mean"] + merged["std"],
                        alpha=0.2,
                    )
            else:
                for run, g in sub.groupby("run"):
                    g_df = cast(pd.DataFrame, g)
                    if xaxis in g_df.columns:
                        g_sorted = g_df.sort_values(xaxis)
                    else:
                        g_sorted = g_df.reset_index(drop=True).assign(**{xaxis: range(len(g_df))})
                    ax.plot(g_sorted[xaxis], g_sorted[value_col], label=str(run))
        ax.set_title(tag)
        ax.set_xlabel(xaxis)
        ax.set_ylabel("value")
        if legend:
            ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot learning curves from TensorBoard event files.")
    p.add_argument("--log-root", nargs="+", required=True, help="イベントファイル探索ルート (複数可)")
    p.add_argument("--tags", nargs="*", default=None, help="抽出する scalar tag (未指定で自動)")
    p.add_argument("--no-recursive", action="store_true", help="再帰探索を無効化")
    p.add_argument("--ema", type=float, default=None, help="EMA スムージング係数 (例: 0.9)")
    p.add_argument("--window", type=int, default=None, help="移動平均ウィンドウ幅")
    p.add_argument("--xaxis", choices=["step", "wall_time"], default="step")
    p.add_argument("--no-aggregate", action="store_true", help="複数 run を平均しない")
    p.add_argument("--no-legend", action="store_true", help="凡例を表示しない")
    p.add_argument("--all-seed-mean", action="store_true", help="全シード(全run)の平均だけを表示 (個別線は非表示)")
    p.add_argument("--group-by-root", action="store_true", help="与えた各log-rootディレクトリを optimizer グループとして分離して平均")
    p.add_argument("--output", type=str, default=None, help="画像保存パス (未指定で表示)")
    # sklearn AUC options
    p.add_argument("--sklearn-auc", action="store_true", help="scikit-learn の auc() で mean 曲線の AUC を計算して表示")
    p.add_argument("--auc-tags", nargs="*", default=None, help="AUC を計算するタグ (未指定なら --tags または自動選択タグ全部)")
    p.add_argument("--auc-output", type=str, default=None, help="AUC 結果CSV 保存パス")
    # wall clock summary
    p.add_argument("--wall-clock-summary", action="store_true", help="wall clock(経過学習時間)タグの最終値平均/標準偏差を表示")
    p.add_argument("--wall-clock-tag", type=str, default="time/training_wall_clock_sec", help="wall clock タグ名")
    p.add_argument("--wall-clock-output", type=str, default=None, help="wall clock 集計CSV 出力")
    return p.parse_args()


def compute_mean_curve_auc(
    df: pd.DataFrame,
    tags: list[str],
    xaxis: str,
    per_group: bool,
) -> pd.DataFrame:
    """Compute AUC (scikit-learn trapezoidal) on the mean-across-runs curve for each tag.

    手順:
        1. 各 tag (と group が有効なら group, tag) で run 別系列を pivot (index=xaxis, columns=run)
        2. 行方向に平均 (NaN 無視) を計算 → mean curve
        3. sklearn.metrics.auc(step, mean_value) で台形積分
    """
    rows: list[dict[str, object]] = []
    have_group = per_group and "group" in df.columns
    tag_set = tags if tags else sorted(df["tag"].unique().astype(str).tolist())
    for tag in tag_set:
        tag_df = df[df.tag == tag]
        if tag_df.empty:
            continue
        if have_group:
            for group_name, g_grp in tag_df.groupby("group"):
                g_grp_df = cast(pd.DataFrame, g_grp)
                auc_val = _single_mean_auc(g_grp_df, xaxis)
                if auc_val is None:
                    continue
                rows.append({"tag": tag, "group": group_name, "auc_mean_curve": auc_val})
        else:
            tag_df_df = cast(pd.DataFrame, tag_df)
            auc_val = _single_mean_auc(tag_df_df, xaxis)
            if auc_val is not None:
                rows.append({"tag": tag, "auc_mean_curve": auc_val})
    return pd.DataFrame(rows)


def _single_mean_auc(sub: pd.DataFrame, xaxis: str) -> float | None:
    # pivot run-wise
    needed_cols = {xaxis, "run", "value"}
    missing = needed_cols - set(sub.columns)
    if missing:
        return None
    # 各 run をソート後 index=step -> value でマージ
    pivot_parts: list[pd.DataFrame] = []
    for run, g in sub.groupby("run"):
        g_sorted = g.sort_values(xaxis)
        g_dedup = g_sorted.drop_duplicates(subset=[xaxis], keep="last")
        if g_dedup.shape[0] < 2:
            continue
        part = g_dedup[[xaxis, "value"]].set_index(xaxis).rename(columns={"value": str(run)})
        pivot_parts.append(part)
    if not pivot_parts:
        return None
    merged = pd.concat(pivot_parts, axis=1)
    # 行平均 (NaN 無視)
    mean_series = merged.mean(axis=1, numeric_only=True)
    steps = mean_series.index.to_numpy(dtype=float)
    vals = mean_series.to_numpy(dtype=float)
    if steps.shape[0] < 2:
        return None
    # sklearn.metrics.auc は x 増加が前提
    order = steps.argsort()
    steps_sorted = steps[order]
    vals_sorted = vals[order]
    try:
        return float(sk_auc(steps_sorted, vals_sorted))
    except Exception:
        return None


def main():
    args = parse_args()
    all_events: list[tuple[Path, str | None]] = []
    for root in args.log_root:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[WARN] path not found: {root}")
            continue
        files = find_event_files(root_path, recursive=not args.no_recursive)
        group_label = root_path.name if args.group_by_root else None
        all_events.extend([(f, group_label) for f in files])
    if not all_events:
        print("[ERROR] No event files found.")
        return

    df = concat_event_scalars(all_events, args.tags)
    if df.empty:
        print("[ERROR] No scalar data found in event files.")
        return

    df = apply_smoothing(df, ema=args.ema, window=args.window)

    fig = plot_tags(
        df,
        xaxis=args.xaxis,
        use_smooth=(args.ema is not None or (args.window is not None and args.window > 1)),
        aggregate_runs=not args.no_aggregate,
        force_aggregate=args.all_seed_mean,
        legend=not args.no_legend,
        per_group=args.group_by_root,
    )

    if fig is None:
        return

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()

    # sklearn AUC (mean curve) calculation
    if args.sklearn_auc:
        target_tags = args.auc_tags if args.auc_tags else (args.tags if args.tags else sorted(df["tag"].unique().astype(str).tolist()))
        auc_df = compute_mean_curve_auc(df, target_tags, args.xaxis, per_group=args.group_by_root)
        if auc_df.empty:
            print("[AUC] No AUC could be computed (insufficient data).")
        else:
            print("[AUC] Mean-curve AUC (sklearn.metrics.auc):")
            print(auc_df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
            if args.auc_output:
                auc_out = Path(args.auc_output)
                auc_out.parent.mkdir(parents=True, exist_ok=True)
                auc_df.to_csv(auc_out, index=False)
                print(f"[AUC] Saved AUC CSV to {auc_out}")

    # Wall clock summary (専用表示) - final_stats とは独立
    if args.wall_clock_summary:
        wc_tag = args.wall_clock_tag
        wc_df = df[df.tag == wc_tag]
        if wc_df.empty:
            print(f"[WALL] Tag '{wc_tag}' not found in loaded data.")
        else:
            have_group = args.group_by_root and "group" in wc_df.columns
            rows: list[dict[str, object]] = []
            group_keys = (["group"] if have_group else []) + ["run"]
            for keys, g in wc_df.groupby(group_keys):
                g_df = cast(pd.DataFrame, g)
                g_sorted = g_df.sort_values("step") if "step" in g_df.columns else g_df
                last_row = g_sorted.iloc[-1]
                rec: dict[str, object] = {
                    "run": last_row["run"],
                    "final_wall_clock_sec": float(last_row["value"]),
                }
                if have_group:
                    rec["group"] = last_row["group"]
                rows.append(rec)
            if not rows:
                print("[WALL] No wall clock data rows.")
            else:
                import pandas as _pd  # type: ignore
                wc_final = _pd.DataFrame(rows)
                agg_cols = (["group"] if have_group else [])
                summary_wc = wc_final.groupby(agg_cols, as_index=False).agg(
                    mean_wall_clock_sec=("final_wall_clock_sec", "mean"),
                    std_wall_clock_sec=("final_wall_clock_sec", "std"),
                    min_wall_clock_sec=("final_wall_clock_sec", "min"),
                    max_wall_clock_sec=("final_wall_clock_sec", "max"),
                    n_runs=("final_wall_clock_sec", "count"),
                )
                print("[WALL] Wall clock final value statistics (sec):")
                print(summary_wc.to_string(index=False))
                if args.wall_clock_output:
                    out_wc = Path(args.wall_clock_output)
                    out_wc.parent.mkdir(parents=True, exist_ok=True)
                    summary_wc.to_csv(out_wc, index=False)
                    print(f"[WALL] Saved wall clock stats CSV to {out_wc}")


if __name__ == "__main__":  # pragma: no cover
    main()
