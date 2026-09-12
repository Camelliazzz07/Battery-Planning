# -*- coding: utf-8 -*-
"""问题3：附件2与附件3的论文用综合探索性分析图。

在 VS Code 中可直接运行。默认读取仓库“附件”目录下的附件2、附件3，
并将一张 SVG 四联图保存到“问题三/辅助输出”。
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ATTACHMENT_DIR = PROJECT_ROOT / "附件"
AUXILIARY_DIR = SCRIPT_DIR / "辅助输出"

COOL = ["#BFDFD2", "#51999F", "#4198AC", "#7BC0CD"]
WARM = ["#DBCB92", "#ECB66C", "#EA9E58", "#ED8D5A"]
LOAD_COLOR = COOL[1]
PV_COLOR = WARM[2]
NET_COLOR = COOL[2]
ERROR_COLOR = WARM[3]

plt.rcParams.update(
    {
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#7A7A7A",
        "axes.labelcolor": "#333333",
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.color": "#4A4A4A",
        "ytick.color": "#4A4A4A",
        "grid.color": "#D9D9D9",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)


def endpoint_minute(value) -> int:
    """将右端点时刻转换为当天累计分钟数，24:00 记为 1440。"""
    if isinstance(value, (datetime, time, pd.Timestamp)):
        minute = value.hour * 60 + value.minute
        return minute or 1440
    if isinstance(value, (float, int, np.floating, np.integer)) and 0 <= value <= 1:
        return int(round(float(value) * 1440)) or 1440
    text = str(value).strip().replace("：", ":")
    if text in {"0:00+1", "00:00+1", "24:00"}:
        return 1440
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match:
        raise ValueError(f"无法解析时间：{value!r}")
    hour, minute = map(int, match.groups())
    if hour > 23 or minute > 59:
        raise ValueError(f"时间超出范围：{value!r}")
    return hour * 60 + minute or 1440


def numeric_matrix(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name}存在缺失、非数值、无穷大或负值。")
    return values


def read_attachment2(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frames = {
        name: pd.read_excel(path, sheet_name=name)
        for name in ("小区负载", "光伏发电实际功率")
    }
    dates: pd.DatetimeIndex | None = None
    arrays: dict[str, np.ndarray] = {}
    expected = np.arange(10, 1441, 10)

    for name, frame in frames.items():
        current_dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0])).normalize()
        if current_dates.has_duplicates:
            raise ValueError(f"附件2工作表“{name}”包含重复日期。")
        if dates is not None and not current_dates.equals(dates):
            raise ValueError("附件2的负荷与光伏日期不一致。")
        dates = current_dates

        minutes = np.array([endpoint_minute(column) for column in frame.columns[1:]])
        order = np.argsort(minutes)
        if not np.array_equal(minutes[order], expected):
            raise ValueError(f"附件2工作表“{name}”缺少或重复十分钟右端点。")
        arrays[name] = numeric_matrix(frame, name)[:, order]

    if dates is None or not dates.equals(pd.date_range(dates[0], dates[-1])):
        raise ValueError("附件2日期不连续。")
    return dates, arrays["小区负载"], arrays["光伏发电实际功率"]


def read_attachment3(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    lead_columns = [f"预报{i}小时" for i in range(1, 25)]
    required = ["日期", "预报时刻", *lead_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"附件3缺少字段：{missing}")

    frame = frame[required].copy()
    frame["日期"] = pd.to_datetime(frame["日期"].ffill()).dt.normalize()
    frame["发布时间"] = frame.apply(
        lambda row: row["日期"]
        + pd.Timedelta(minutes=endpoint_minute(row["预报时刻"]) % 1440),
        axis=1,
    )
    hours = frame["发布时间"].dt.hour
    if not hours.isin([0, 6, 12, 18]).all():
        raise ValueError("附件3包含预期之外的预报时刻。")
    frame[lead_columns] = frame[lead_columns].apply(pd.to_numeric, errors="coerce")
    values = frame[lead_columns].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("附件3预测值存在缺失、非数值、无穷大或负值。")
    if frame.duplicated(["日期", "预报时刻"]).any():
        raise ValueError("附件3包含重复的预报发布记录。")
    return frame


def hourly_forecast_errors(
    forecasts: pd.DataFrame,
    dates: pd.DatetimeIndex,
    actual_pv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将未来1—24小时预测与附件2相同右端点的实际光伏功率匹配。"""
    actual_index = [
        day + pd.Timedelta(minutes=int(minute))
        for day in dates
        for minute in np.arange(10, 1441, 10)
    ]
    actual = pd.Series(actual_pv.ravel(), index=pd.DatetimeIndex(actual_index))
    lead_columns = [f"预报{i}小时" for i in range(1, 25)]
    records: list[tuple[int, float]] = []

    for _, row in forecasts.iterrows():
        issue = row["发布时间"]
        for lead, column in enumerate(lead_columns, start=1):
            target = issue + pd.Timedelta(hours=lead)
            if target in actual.index:
                records.append((lead, float(row[column]) - float(actual.loc[target])))

    if not records:
        raise ValueError("附件3预测时刻无法与附件2实际光伏数据匹配。")
    errors = pd.DataFrame(records, columns=["提前期", "误差"])
    grouped = errors.groupby("提前期")["误差"]
    leads = np.arange(1, 25)
    mae = grouped.apply(lambda values: values.abs().mean()).reindex(leads).to_numpy()
    bias = grouped.mean().reindex(leads).to_numpy()
    q90 = grouped.apply(lambda values: values.abs().quantile(0.90)).reindex(leads).to_numpy()
    counts = grouped.size().reindex(leads, fill_value=0).to_numpy()
    return leads, mae, bias, q90, counts


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_overview(
    dates: pd.DatetimeIndex,
    load: np.ndarray,
    pv: np.ndarray,
    forecasts: pd.DataFrame,
    output: Path,
) -> None:
    net = load - pv
    daily_load, daily_pv = load.mean(axis=1), pv.mean(axis=1)
    minutes = np.arange(10, 1441, 10)
    hours = minutes / 60
    leads, mae, bias, q90, counts = hourly_forecast_errors(forecasts, dates, pv)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.plot(dates, daily_load, color=LOAD_COLOR, lw=1.55, label="日均负荷")
    ax.plot(dates, daily_pv, color=PV_COLOR, lw=1.55, label="日均光伏")
    ax.set_title("(a) 全年负荷与光伏日均功率")
    ax.set_ylabel("功率（kW）")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax)

    ax = axes[0, 1]
    for values, color, label in (
        (load, LOAD_COLOR, "负荷"),
        (pv, PV_COLOR, "光伏"),
    ):
        lower, median, upper = np.quantile(values, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(hours, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.plot(hours, median, color=color, lw=2.0, label=f"{label}中位数")
    ax.set_title("(b) 典型日内曲线及10%—90%区间")
    ax.set_xlabel("时刻")
    ax.set_ylabel("功率（kW）")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.legend(loc="upper right")
    style_axis(ax)

    ax = axes[1, 0]
    daily_net_peak = net.max(axis=1)
    monthly = [daily_net_peak[dates.month == month] for month in range(1, 13)]
    box = ax.boxplot(
        monthly,
        positions=np.arange(1, 13),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#FFFFFF", "linewidth": 1.5},
        whiskerprops={"color": NET_COLOR, "linewidth": 1.0},
        capprops={"color": NET_COLOR, "linewidth": 1.0},
        boxprops={"edgecolor": NET_COLOR, "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(NET_COLOR)
        patch.set_alpha(0.78)
    ax.set_title("(c) 各月日最大净负荷分布")
    ax.set_xlabel("月份")
    ax.set_ylabel("日最大净负荷（kW）")
    ax.set_xticks(np.arange(1, 13), [str(month) for month in range(1, 13)])
    style_axis(ax)

    ax = axes[1, 1]
    ax.fill_between(leads, mae, q90, color=ERROR_COLOR, alpha=0.14, label="MAE至90%绝对误差")
    ax.plot(leads, mae, color=ERROR_COLOR, lw=2.0, marker="o", ms=3.6, label="MAE")
    ax.plot(leads, bias, color=LOAD_COLOR, lw=1.8, marker="s", ms=3.2, label="平均偏差")
    ax.axhline(0, color="#666666", lw=0.9, alpha=0.75)
    ax.set_title("(d) 光伏预测误差随提前期变化")
    ax.set_xlabel("预测提前期（小时）")
    ax.set_ylabel("预测误差（kW）")
    ax.set_xlim(1, 24)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="upper left")
    style_axis(ax)

    fig.suptitle("附件2与附件3数据特征综合分析", fontsize=16, fontweight="semibold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"数据范围：{dates[0].date()} 至 {dates[-1].date()}，共 {len(dates)} 天")
    print(f"附件3有效预测误差样本：{int(counts.sum())} 个")
    print(f"光伏预测总体 MAE：{np.average(mae, weights=counts):.2f} kW")
    print(f"综合图已保存：{output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成问题3附件2与附件3综合EDA图")
    parser.add_argument("--attachment2", type=Path, default=ATTACHMENT_DIR / "附件2.xlsx")
    parser.add_argument("--attachment3", type=Path, default=ATTACHMENT_DIR / "附件3.xlsx")
    parser.add_argument(
        "--output",
        type=Path,
        default=AUXILIARY_DIR / "问题3_附件2与附件3综合EDA.svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".svg":
        raise ValueError("输出文件必须使用 .svg 扩展名。")
    dates, load, pv = read_attachment2(args.attachment2)
    forecasts = read_attachment3(args.attachment3)
    plot_overview(dates, load, pv, forecasts, args.output)


if __name__ == "__main__":
    main()
