#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题1：附件1数据预处理、探索性数据分析（EDA）与空白结果表生成。

运行示例：
    python question1_preprocess_eda.py --input 附件1.xlsx --output-dir question1_output

时间约定：附件1采用右端点记时。原始时刻 0:10 表示区间 0:00-0:10，
原始时刻 0:00+1 表示区间 23:50-24:00。
"""

from __future__ import annotations

import argparse
import re
import warnings
from datetime import datetime, time
from pathlib import Path
from typing import Any

import matplotlib

# 使脚本在 VS Code 终端、“运行 Python 文件”和无图形界面环境下均能稳定保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ------------------------------ 全局配置 ------------------------------
DT_HOURS = 1.0 / 6.0
EXPECTED_ROWS = 144
EXPECTED_STEP_MINUTES = 10
SCRIPT_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS = ["时间", "电价", "小区负载", "光伏发电预测功率"]
NUMERIC_COLUMNS = ["电价", "小区负载", "光伏发电预测功率"]

sns.set_theme(style="whitegrid")

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# -------------------------- 模块1：读取与时间解析 --------------------------
def parse_time_to_minutes(value: Any) -> int:
    """把附件时间转换为从当天 0:00 起累计的分钟数。"""
    if pd.isna(value):
        raise ValueError("时间列中存在空值，无法建立144个固定时段。")

    if isinstance(value, pd.Timestamp):
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, time):
        return value.hour * 60 + value.minute

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if 0 <= number <= 1:
            return int(round(number * 24 * 60))
        raise ValueError(f"无法识别的数值型时间：{value!r}")

    text = str(value).strip().replace("：", ":")
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(\+1)?", text)
    if m is None:
        raise ValueError(f"无法解析时间：{value!r}")

    hour = int(m.group(1))
    minute = int(m.group(2))
    next_day = m.group(3) is not None

    if hour > 23 or minute > 59:
        raise ValueError(f"时间超出合法范围：{value!r}")
    if next_day and not (hour == 0 and minute == 0):
        raise ValueError(f"仅允许 0:00+1 表示次日零点，收到：{value!r}")

    return 1440 if next_day else hour * 60 + minute


def format_minute(minute: int, pad_hour: bool = True) -> str:
    """将累计分钟格式化；跨日时使用 0:00+1、0:10+1 等形式。"""
    if minute < 0:
        raise ValueError("累计分钟数不能为负数。")

    day_offset, minute_in_day = divmod(int(minute), 1440)
    hour, minute_part = divmod(minute_in_day, 60)
    hour_text = f"{hour:02d}" if pad_hour and day_offset == 0 else str(hour)
    suffix = f"+{day_offset}" if day_offset else ""
    return f"{hour_text}:{minute_part:02d}{suffix}"


def warn_about_time_axis(minutes: pd.Series) -> list[str]:
    """检查行数、重复时刻、10分钟步长和完整的右端点序列。"""
    messages: list[str] = []

    if len(minutes) != EXPECTED_ROWS:
        messages.append(
            f"行数异常：实际 {len(minutes)} 行，预期 {EXPECTED_ROWS} 行。"
        )

    duplicate_values = minutes[minutes.duplicated(keep=False)].tolist()
    if duplicate_values:
        messages.append(f"发现重复时刻（分钟）：{sorted(set(duplicate_values))}")

    differences = minutes.diff().dropna()
    bad_steps = differences[differences != EXPECTED_STEP_MINUTES]
    if not bad_steps.empty:
        detail = [
            f"第{idx}至第{idx + 1}个排序点：{int(step)}分钟" # type: ignore
            for idx, step in bad_steps.items()
        ]
        messages.append("发现非10分钟步长：" + "；".join(detail))

    expected = np.arange(10, 1441, 10)
    actual = minutes.to_numpy(dtype=int)
    if len(actual) == len(expected) and not np.array_equal(actual, expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        messages.append(
            f"时间轴并非完整的 10,20,...,1440；缺失={missing}，额外={extra}。"
        )

    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return messages


def load_and_parse_data(input_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """读取附件1，解析并排序时间，生成右端点对应的区间信息。"""
    print(f"[读取] {input_path.resolve()}")
    df = pd.read_excel(input_path, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise KeyError(f"缺少必要列：{missing_columns}；实际列为：{df.columns.tolist()}")

    df = df[REQUIRED_COLUMNS].copy()
    df["时间分钟"] = df["时间"].map(parse_time_to_minutes)
    df = df.sort_values("时间分钟", kind="stable").reset_index(drop=True)

    time_messages = warn_about_time_axis(df["时间分钟"])
    df.insert(0, "区间编号", np.arange(1, len(df) + 1, dtype=int))
    df["区间起点"] = (df["时间分钟"] - EXPECTED_STEP_MINUTES).map(format_minute)
    df["区间终点"] = df["时间分钟"].map(format_minute)

    return df, time_messages


# ---------------------- 模块2：数据清洗与派生变量 ----------------------
def clean_and_derive(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """清洗数值列并计算能量及净负荷，不改变时间行数。"""
    report: list[str] = []
    cleaned = df.copy()

    for column in NUMERIC_COLUMNS:
        original_missing = int(cleaned[column].isna().sum())
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        after_coercion_missing = int(cleaned[column].isna().sum())
        newly_invalid = after_coercion_missing - original_missing

        if newly_invalid > 0:
            report.append(
                f"{column}：有 {newly_invalid} 个非数值内容被转换为 NaN。"
            )

        if after_coercion_missing > 0:
            cleaned[column] = cleaned[column].interpolate(
                method="linear", limit_direction="both"
            )
            remaining = int(cleaned[column].isna().sum())
            report.append(
                f"{column}：发现 {after_coercion_missing} 个缺失值，"
                f"已按时间顺序进行线性插值；剩余 {remaining} 个。"
            )
        else:
            report.append(f"{column}：无缺失值。")

        if cleaned[column].isna().any():
            raise ValueError(f"{column} 仍存在无法填补的缺失值，请人工检查。")

    cleaned["负载电量_kWh"] = cleaned["小区负载"] * DT_HOURS
    cleaned["光伏电量_kWh"] = cleaned["光伏发电预测功率"] * DT_HOURS
    cleaned["净负荷_kW"] = cleaned["小区负载"] - cleaned["光伏发电预测功率"]

    report.append(f"时间间隔 dt = {DT_HOURS:.6f} 小时（10分钟）。")
    report.append("派生变量计算完成：负载电量、光伏电量、净负荷功率。")
    return cleaned, report


# -------------------------- 模块3：可视化 EDA --------------------------
def plot_combined_time_series(df: pd.DataFrame, output_path: Path) -> None:
    """输出负载、光伏、净负荷与阶梯电价的双Y轴综合图。"""
    fig, ax_left = plt.subplots(figsize=(15, 7.5))
    x = df["区间编号"].to_numpy()

    ax_left.plot(x, df["小区负载"], color="#2563EB", lw=2.0, label="小区负载 (kW)")
    ax_left.plot(
        x,
        df["光伏发电预测功率"],
        color="#F59E0B",
        lw=2.0,
        label="光伏发电预测功率 (kW)",
    )
    ax_left.plot(x, df["净负荷_kW"], color="#16A34A", lw=2.0, label="净负荷 (kW)")
    ax_left.set_xlabel("区间编号（1-144）")
    ax_left.set_ylabel("功率 (kW)")
    ax_left.set_xlim(1, len(df))
    ax_left.grid(True, which="major", alpha=0.32, linestyle="--")

    ax_right = ax_left.twinx()
    ax_right.step(
        x,
        df["电价"],
        where="mid",
        color="#DC2626",
        lw=1.8,
        alpha=0.9,
        label="电价 (元/kWh)",
    )
    ax_right.set_ylabel("电价 (元/kWh)", color="#B91C1C")
    ax_right.tick_params(axis="y", labelcolor="#B91C1C")

    tick_positions = np.arange(1, len(df) + 1, 12)
    ax_left.set_xticks(tick_positions)
    ax_left.set_xticklabels(df.loc[tick_positions - 1, "区间终点"], rotation=0)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=4,
        frameon=True,
    )
    ax_left.set_title("问题1：负载、光伏、净负荷与电价的时间对齐")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[绘图] 已保存：{output_path}")


def plot_net_load_regions(df: pd.DataFrame, output_path: Path) -> None:
    """输出净负荷正负区域填充图。"""
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = df["区间编号"].to_numpy(dtype=float)
    net_load = df["净负荷_kW"].to_numpy(dtype=float)

    ax.plot(x, net_load, color="#334155", lw=2.1, label="净负荷 (kW)")
    
    ax.fill_between(
        x,
        0,
        net_load,
        where=(net_load > 0).tolist(),
        interpolate=True,
        color="#EF4444",
        alpha=0.35,
        label="净负荷 > 0（电力缺口）",
    )
    ax.fill_between(
        x,
        0,
        net_load,
        where=(net_load < 0).tolist(),
        interpolate=True,
        color="#22C55E",
        alpha=0.38,
        label="净负荷 < 0（光伏富余）",
    )
    ax.axhline(0, color="black", lw=1.0, alpha=0.75)
    ax.set_title("问题1：净负荷正负分区")
    ax.set_xlabel("区间编号（1-144）")
    ax.set_ylabel("净负荷 (kW)")
    ax.set_xlim(1, len(df))
    ax.grid(True, alpha=0.32, linestyle="--")

    tick_positions = np.arange(1, len(df) + 1, 12)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(df.loc[tick_positions - 1, "区间终点"])
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[绘图] 已保存：{output_path}")


# ----------------------- 模块4：模板映射与输出 -----------------------
def build_blank_result_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """按 result1.xlsx 模板的首尾对调逻辑生成144行空白结果表。"""
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"生成标准模板需要 {EXPECTED_ROWS} 行数据，当前为 {len(df)} 行。"
        )

    source_point_order = list(range(2, EXPECTED_ROWS + 1)) + [1]
    interval_starts = np.arange(10, 1450, 10)
    
    interval_labels = [
        f"{format_minute(int(start), pad_hour=False)}-{format_minute(int(start) + 10, pad_hour=False)}"
        for start in interval_starts
    ]

    result = pd.DataFrame(
        {
            "时间段": interval_labels,
            "购电量": pd.Series([np.nan] * EXPECTED_ROWS, dtype="float64"),
        }
    )
    return result, source_point_order


def save_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    """保存清洗数据、空白结果表与两张核心EDA图片。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = output_dir / "附件1_clean.csv"
    blank_result_xlsx = output_dir / "result1_blank.xlsx"
    combined_png = output_dir / "问题1_时间序列综合图.png"
    net_load_png = output_dir / "问题1_净负荷分区图.png"

    df.to_csv(clean_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 已保存：{clean_csv}")

    result_table, source_order = build_blank_result_table(df)
    with pd.ExcelWriter(blank_result_xlsx, engine="openpyxl") as writer:
        result_table.to_excel(writer, sheet_name="计划购电量", index=False)
    print(f"[输出] 已保存：{blank_result_xlsx}")

    print("\n========== 模板映射核验 ==========")
    print(
        f"模板首行：{result_table.loc[0, '时间段']} -> "
        f"附件第{source_order[0]}点（原始时刻 {df.loc[source_order[0]-1, '时间']}）"
    )
    print(
        f"模板第2行：{result_table.loc[1, '时间段']} -> "
        f"附件第{source_order[1]}点（原始时刻 {df.loc[source_order[1]-1, '时间']}）"
    )
    print(
        f"模板末行：{result_table.loc[143, '时间段']} -> "
        f"附件第{source_order[-1]}点（原始时刻 {df.loc[source_order[-1]-1, '时间']}）"
    )

    plot_combined_time_series(df, combined_png)
    plot_net_load_regions(df, net_load_png)


def print_summary(
    df: pd.DataFrame, time_messages: list[str], cleaning_report: list[str]
) -> None:
    """打印时间检查、清洗日志和描述性统计。"""
    print("\n========== 时间轴检查 ==========")
    if time_messages:
        for message in time_messages:
            print(f"[警告] {message}")
    else:
        print("[通过] 共144行；无重复；时间为10,20,...,1440分钟；步长均为10分钟。")

    print("\n========== 清洗报告 ==========")
    for message in cleaning_report:
        print(f"- {message}")

    print("\n========== 数据摘要 ==========")
    summary_columns = NUMERIC_COLUMNS + ["负载电量_kWh", "光伏电量_kWh", "净负荷_kW"]
    print(df[summary_columns].describe().round(4).to_string())

    print("\n========== 能量汇总 ==========")
    print(f"全天负载电量：{df['负载电量_kWh'].sum():.4f} kWh")
    print(f"全天预测光伏电量：{df['光伏电量_kWh'].sum():.4f} kWh")
    print(f"全天净负荷电量：{(df['净负荷_kW'] * DT_HOURS).sum():.4f} kWh")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题1：附件1预处理与EDA")
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "附件1.xlsx",
        help="附件1.xlsx 的路径（默认：脚本所在目录/附件1.xlsx）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="输出目录（默认：脚本所在的“问题一”目录）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件：{args.input.resolve()}")

    df, time_messages = load_and_parse_data(args.input)
    df, cleaning_report = clean_and_derive(df)
    print_summary(df, time_messages, cleaning_report)
    save_outputs(df, args.output_dir)
    print("\n[完成] 问题1的数据预处理、EDA与空白结果表生成完毕。")


if __name__ == "__main__":
    main()
