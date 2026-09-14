from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ANALYSIS_RESULTS_DIRECTORY
from src.utils.resolve_path import resolve_project_path

QUALITY_COLUMNS = {
    "retrieval": [
        "context_precision",
        "context_recall",
        "context_relevance",
    ],
    "answer": [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ],
}

TIME_COLUMNS = [
    "retrieval_time",
    "translation_time",
    "generate_answer_time",
    "total_time",
]

HARDWARE_COLUMNS = [
    "avg_ram_mb",
    "peak_ram_mb",
    "avg_cpu_percent",
    "peak_cpu_percent",
]

ERROR_COLUMNS = [
    "error_phase",
    "error_type",
    "error_message",
    "error_status_code",
]


def normalize_method(method: str) -> str:
    method = str(method).strip().lower()

    mapping = {
        "trag": "tRAG",
        "tr ag": "tRAG",
        "tr": "tRAG",
        "monorag": "MonoRAG",
        "mono": "MonoRAG",
        "multirag": "MultiRAG",
        "multi": "MultiRAG",
        "crossrag": "CrossRAG",
        "cross": "CrossRAG",
    }

    return mapping.get(method, method)


def safe_name(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^\w\-]+", "_", value)
    return value.strip("_")


def create_directories(base_output: Path) -> dict:
    directories = {
        "root": base_output,
        "tables": base_output / "tables",
        "figures": base_output / "figures",
        "datasets": base_output / "datasets",
        "methods": base_output / "methods",
        "comparison": base_output / "comparison",
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    plt.close(fig)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def load_input_file(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input-Datei nicht gefunden: {path}")

    df = pd.read_csv(path)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def load_hardware_file(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Hardware-Datei nicht gefunden: {path}")

    df = pd.read_csv(path)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = (
            QUALITY_COLUMNS["retrieval"]
            + QUALITY_COLUMNS["answer"]
            + TIME_COLUMNS
            + HARDWARE_COLUMNS
            + ["error_status_code"]
    )

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def calculate_summary(
        df: pd.DataFrame,
        columns: list[str],
) -> pd.DataFrame:

    existing_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not existing_columns:
        return pd.DataFrame()

    summary = pd.DataFrame(
        {
            "metric": existing_columns,
            "count": [
                df[column].count()
                for column in existing_columns
            ],
            "mean": [
                df[column].mean()
                for column in existing_columns
            ],
            "median": [
                df[column].median()
                for column in existing_columns
            ],
            "std": [
                df[column].std()
                for column in existing_columns
            ],
            "min": [
                df[column].min()
                for column in existing_columns
            ],
            "max": [
                df[column].max()
                for column in existing_columns
            ],
        }
    )

    return summary


def calculate_quality_summary(
        df: pd.DataFrame,
        columns: list[str],
) -> pd.DataFrame:

    existing_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not existing_columns:
        return pd.DataFrame()

    result = []

    for column in existing_columns:
        values = df[column].dropna().to_numpy(dtype=float)

        if len(values) == 0:
            result.append(
                {
                    "metric": column,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )

            continue

        result.append(
            {
                "metric": column,
                "n": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values, ddof=1)
                if len(values) > 1
                else 0.0,
                "min": np.min(values),
                "max": np.max(values),
            }
        )

    return pd.DataFrame(result)


def calculate_error_analysis(df: pd.DataFrame) -> dict:
    total = len(df)

    if total == 0:
        return {
            "summary": pd.DataFrame(),
            "phase": pd.DataFrame(),
            "type": pd.DataFrame(),
            "description": pd.DataFrame(),
        }

    if "error_phase" not in df.columns:
        error_mask = pd.Series(False, index=df.index)
    else:
        error_mask = (
            df["error_phase"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

    error_count = int(error_mask.sum())
    successful_count = total - error_count

    summary = pd.DataFrame(
        [
            {
                "total_questions": total,
                "errors": error_count,
                "successful": successful_count,
                "error_rate": error_count / total,
                "success_rate": successful_count / total,
            }
        ]
    )

    def categorical_analysis(column: str) -> pd.DataFrame:
        if column not in df.columns:
            return pd.DataFrame()

        values = (
            df.loc[error_mask, column]
            .fillna("Unknown")
            .astype(str)
            .replace("", "Unknown")
        )

        if values.empty:
            return pd.DataFrame()

        result = (
            values.value_counts()
            .rename_axis(column)
            .reset_index(name="count")
        )

        result["percentage_of_errors"] = (
                result["count"] / error_count
        )

        return result

    return {
        "summary": summary,
        "phase": categorical_analysis("error_phase"),
        "type": categorical_analysis("error_type"),
        "description": categorical_analysis("error_message"),
    }


def calculate_dataset_statistics(
        df: pd.DataFrame,
) -> pd.DataFrame:

    statistics = {
        "questions": len(df),
    }

    for column in QUALITY_COLUMNS["retrieval"]:
        if column in df.columns:
            statistics[f"{column}_mean"] = df[column].mean()

    for column in QUALITY_COLUMNS["answer"]:
        if column in df.columns:
            statistics[f"{column}_mean"] = df[column].mean()

    for column in TIME_COLUMNS:
        if column in df.columns:
            statistics[f"{column}_mean"] = df[column].mean()

    for column in HARDWARE_COLUMNS:
        if column in df.columns:
            statistics[f"{column}_mean"] = df[column].mean()

    error_analysis = calculate_error_analysis(df)

    if not error_analysis["summary"].empty:
        row = error_analysis["summary"].iloc[0]
        statistics["errors"] = row["errors"]
        statistics["error_rate"] = row["error_rate"]

    return pd.DataFrame([statistics])


def merge_input_and_hardware(
        input_df: pd.DataFrame,
        hardware_df: pd.DataFrame,
) -> pd.DataFrame:

    input_df = input_df.copy()
    hardware_df = hardware_df.copy()

    input_df["id"] = input_df["id"].astype(str)

    hardware_df["question_id"] = (
        hardware_df["question_id"].astype(str)
    )

    hardware_columns = [
        "avg_ram_mb",
        "peak_ram_mb",
        "avg_cpu_percent",
        "peak_cpu_percent",
    ]

    existing_hardware_columns = [
        column
        for column in hardware_columns
        if column in hardware_df.columns
    ]

    aggregation = {
        column: "mean"
        for column in existing_hardware_columns
    }

    if aggregation:
        hardware_df = (
            hardware_df
            .groupby("question_id", as_index=False)
            .agg(aggregation)
        )

    hardware_df = hardware_df.rename(
        columns={
            "question_id": "id"
        }
    )

    merged = input_df.merge(
        hardware_df,
        on="id",
        how="left",
    )

    return merged


def calculate_phase_hardware_analysis(
        hardware_df: pd.DataFrame,
        method: str | None = None,
) -> pd.DataFrame:

    required_columns = [
        "question_id",
        "phase",
        "avg_ram_mb",
        "peak_ram_mb",
        "avg_cpu_percent",
        "peak_cpu_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in hardware_df.columns
    ]

    if missing_columns:
        return pd.DataFrame(
            columns=[
                "method",
                "phase",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_cpu_percent",
                "peak_cpu_percent",
            ]
        )

    df = hardware_df.copy()

    df["question_id"] = df["question_id"].astype(str)

    df["phase"] = (
        df["phase"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df["phase"] = df["phase"].replace(
        "",
        "Unknown",
    )

    for column in HARDWARE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "avg_ram_mb",
            "peak_ram_mb",
            "avg_cpu_percent",
            "peak_cpu_percent",
        ],
        how="all",
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "phase",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_cpu_percent",
                "peak_cpu_percent",
            ]
        )

    result = (
        df.groupby(
            "phase",
            as_index=False,
        )[HARDWARE_COLUMNS]
        .mean()
    )

    if method is not None:
        result.insert(
            0,
            "method",
            normalize_method(method),
        )

    return result


def combine_phase_hardware_by_method(
        datasets: list[dict],
) -> pd.DataFrame:

    rows = []

    for item in datasets:
        hardware_path = item["hardware"]

        if not Path(hardware_path).exists():
            continue

        hardware_df = load_hardware_file(
            hardware_path
        )

        hardware_df = convert_numeric_columns(
            hardware_df
        )

        method = normalize_method(
            item["method"]
        )

        required_columns = [
            "question_id",
            "phase",
            "avg_ram_mb",
            "peak_ram_mb",
            "avg_cpu_percent",
            "peak_cpu_percent",
        ]

        if not all(
                column in hardware_df.columns
                for column in required_columns
        ):
            continue

        hardware_df["method"] = method

        rows.append(
            hardware_df[
                [
                    "question_id",
                    "phase",
                    "avg_ram_mb",
                    "peak_ram_mb",
                    "avg_cpu_percent",
                    "peak_cpu_percent",
                    "method",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "phase",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_cpu_percent",
                "peak_cpu_percent",
            ]
        )

    combined = pd.concat(
        rows,
        ignore_index=True,
    )

    combined["phase"] = (
        combined["phase"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace(
            "",
            "Unknown",
        )
    )

    result = (
        combined
        .groupby(
            ["method", "phase"],
            as_index=False,
        )[HARDWARE_COLUMNS]
        .mean()
    )

    return result


def plot_phase_hardware_ram(
        phase_df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    if phase_df.empty or "phase" not in phase_df.columns:
        return

    plot_data = phase_df.dropna(
        subset=["avg_ram_mb"]
    ).copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    sns.barplot(
        data=plot_data,
        x="phase",
        y="avg_ram_mb",
        hue="method"
        if "method" in plot_data.columns
        else None,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(
        "Durchschnittlicher RAM-Verbrauch (MB)"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    if "method" in plot_data.columns:
        ax.legend(
            title="Methode",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_phase_hardware_peak_ram(
        phase_df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    if phase_df.empty or "phase" not in phase_df.columns:
        return

    plot_data = phase_df.dropna(
        subset=["peak_ram_mb"]
    ).copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    sns.barplot(
        data=plot_data,
        x="phase",
        y="peak_ram_mb",
        hue="method"
        if "method" in plot_data.columns
        else None,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(
        "Durchschnittlicher Peak-RAM (MB)"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    if "method" in plot_data.columns:
        ax.legend(
            title="Methode",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_phase_hardware_cpu(
        phase_df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    if phase_df.empty or "phase" not in phase_df.columns:
        return

    plot_data = phase_df.dropna(
        subset=["avg_cpu_percent"]
    ).copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    sns.barplot(
        data=plot_data,
        x="phase",
        y="avg_cpu_percent",
        hue="method"
        if "method" in plot_data.columns
        else None,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(
        "Durchschnittliche CPU-Auslastung (%)"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    if "method" in plot_data.columns:
        ax.legend(
            title="Methode",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_phase_hardware_peak_cpu(
        phase_df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    if phase_df.empty or "phase" not in phase_df.columns:
        return

    plot_data = phase_df.dropna(
        subset=["peak_cpu_percent"]
    ).copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    sns.barplot(
        data=plot_data,
        x="phase",
        y="peak_cpu_percent",
        hue="method"
        if "method" in plot_data.columns
        else None,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(
        "Durchschnittlicher Peak-CPU-Wert (%)"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    if "method" in plot_data.columns:
        ax.legend(
            title="Methode",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def save_phase_hardware_analysis(
        hardware_df: pd.DataFrame,
        method: str,
        output_table_dir: Path,
        output_figure_dir: Path,
        title_prefix: str,
) -> pd.DataFrame:

    phase_df = calculate_phase_hardware_analysis(
        hardware_df=hardware_df,
        method=method,
    )

    if phase_df.empty:
        return phase_df

    save_table(
        phase_df,
        output_table_dir / "hardware_by_phase.csv",
        )

    plot_phase_hardware_ram(
        phase_df,
        f"{title_prefix} – Ø RAM nach Phase",
        output_figure_dir
        / "hardware_avg_ram_by_phase",
        )

    plot_phase_hardware_peak_ram(
        phase_df,
        f"{title_prefix} – Peak RAM nach Phase",
        output_figure_dir
        / "hardware_peak_ram_by_phase",
        )

    plot_phase_hardware_cpu(
        phase_df,
        f"{title_prefix} – Ø CPU nach Phase",
        output_figure_dir
        / "hardware_avg_cpu_by_phase",
        )

    plot_phase_hardware_peak_cpu(
        phase_df,
        f"{title_prefix} – Peak CPU nach Phase",
        output_figure_dir
        / "hardware_peak_cpu_by_phase",
        )

    return phase_df


def plot_method_phase_hardware_comparison(
        phase_df: pd.DataFrame,
        output_dir: Path,
) -> None:

    if phase_df.empty:
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_phase_hardware_ram(
        phase_df,
        "Vergleich des Ø RAM-Verbrauchs nach Methode und Phase",
        output_dir
        / "methods_hardware_avg_ram_by_phase",
        )

    plot_phase_hardware_peak_ram(
        phase_df,
        "Vergleich des Peak-RAM-Verbrauchs nach Methode und Phase",
        output_dir
        / "methods_hardware_peak_ram_by_phase",
        )

    plot_phase_hardware_cpu(
        phase_df,
        "Vergleich der Ø CPU-Auslastung nach Methode und Phase",
        output_dir
        / "methods_hardware_avg_cpu_by_phase",
        )

    plot_phase_hardware_peak_cpu(
        phase_df,
        "Vergleich der Peak-CPU-Auslastung nach Methode und Phase",
        output_dir
        / "methods_hardware_peak_cpu_by_phase",
        )


def plot_quality(
        df: pd.DataFrame,
        columns: list[str],
        title: str,
        output_path: Path,
) -> None:

    existing = [
        column
        for column in columns
        if column in df.columns
    ]

    if not existing:
        return

    plot_data = df[existing].copy()

    renamed = {
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "context_relevance": "Context Relevancy",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
    }

    plot_data = plot_data.rename(
        columns=renamed
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.boxplot(
        data=plot_data,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metrik")
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_quality_means(
        df: pd.DataFrame,
        columns: list[str],
        title: str,
        output_path: Path,
) -> None:

    existing = [
        column
        for column in columns
        if column in df.columns
    ]

    if not existing:
        return

    values = [
        df[column].mean()
        for column in existing
    ]

    labels = [
        {
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
            "context_relevance": "Context Relevancy",
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "answer_correctness": "Answer Correctness",
        }.get(column, column)
        for column in existing
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.barplot(
        x=labels,
        y=values,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("Mittlerer Score")
    ax.set_xlabel("Metrik")
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    for index, value in enumerate(values):
        if not np.isnan(value):
            ax.text(
                index,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_times(
        df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    existing = [
        column
        for column in TIME_COLUMNS
        if column in df.columns
    ]

    if not existing:
        return

    plot_data = df[existing].copy()

    renamed = {
        "retrieval_time": "Retrieval",
        "translation_time": "Translation",
        "generate_answer_time": "Generation",
        "total_time": "Gesamt",
    }

    plot_data = plot_data.rename(
        columns=renamed
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.boxplot(
        data=plot_data,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("Zeit")
    ax.set_xlabel("Messgröße")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_time_means(
        df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    existing = [
        column
        for column in TIME_COLUMNS
        if column in df.columns
    ]

    if not existing:
        return

    labels = [
        {
            "retrieval_time": "Retrieval",
            "translation_time": "Translation",
            "generate_answer_time": "Generation",
            "total_time": "Gesamt",
        }.get(column, column)
        for column in existing
    ]

    values = [
        df[column].mean()
        for column in existing
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.barplot(
        x=labels,
        y=values,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("Mittlere Zeit")
    ax.set_xlabel("Messgröße")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    for index, value in enumerate(values):
        if not np.isnan(value):
            ax.text(
                index,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_hardware(
        df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    existing = [
        column
        for column in HARDWARE_COLUMNS
        if column in df.columns
    ]

    if not existing:
        return

    plot_data = df[existing].copy()

    renamed = {
        "avg_ram_mb": "Ø RAM",
        "peak_ram_mb": "Peak RAM",
        "avg_cpu_percent": "Ø CPU",
        "peak_cpu_percent": "Peak CPU",
    }

    plot_data = plot_data.rename(
        columns=renamed
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10),
    )

    ram_columns = [
        column
        for column in ["Ø RAM", "Peak RAM"]
        if column in plot_data.columns
    ]

    cpu_columns = [
        column
        for column in ["Ø CPU", "Peak CPU"]
        if column in plot_data.columns
    ]

    if ram_columns:
        sns.boxplot(
            data=plot_data[ram_columns],
            ax=axes[0],
        )

        axes[0].set_title("RAM-Nutzung")
        axes[0].set_ylabel("MB")
        axes[0].set_xlabel("Messgröße")

    else:
        axes[0].set_visible(False)

    if cpu_columns:
        sns.boxplot(
            data=plot_data[cpu_columns],
            ax=axes[1],
        )

        axes[1].set_title("CPU-Nutzung")
        axes[1].set_ylabel("Prozent")
        axes[1].set_xlabel("Messgröße")

    else:
        axes[1].set_visible(False)

    fig.suptitle(title)

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        fig,
        output_path,
    )


def plot_hardware_means(
        df: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    existing = [
        column
        for column in HARDWARE_COLUMNS
        if column in df.columns
    ]

    if not existing:
        return

    ram_columns = [
        column
        for column in ["avg_ram_mb", "peak_ram_mb"]
        if column in df.columns
    ]

    cpu_columns = [
        column
        for column in ["avg_cpu_percent", "peak_cpu_percent"]
        if column in df.columns
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10),
    )

    if ram_columns:
        labels = [
                     "Ø RAM",
                     "Peak RAM",
                 ][:len(ram_columns)]

        values = [
            df[column].mean()
            for column in ram_columns
        ]

        sns.barplot(
            x=labels,
            y=values,
            ax=axes[0],
        )

        axes[0].set_title(
            "Mittlere RAM-Nutzung"
        )
        axes[0].set_ylabel("MB")

        for index, value in enumerate(values):
            axes[0].text(
                index,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    else:
        axes[0].set_visible(False)

    if cpu_columns:
        labels = [
                     "Ø CPU",
                     "Peak CPU",
                 ][:len(cpu_columns)]

        values = [
            df[column].mean()
            for column in cpu_columns
        ]

        sns.barplot(
            x=labels,
            y=values,
            ax=axes[1],
        )

        axes[1].set_title(
            "Mittlere CPU-Nutzung"
        )
        axes[1].set_ylabel("Prozent")

        for index, value in enumerate(values):
            axes[1].text(
                index,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    else:
        axes[1].set_visible(False)

    fig.suptitle(title)

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        fig,
        output_path,
    )


def plot_error_rate(
        error_summary: pd.DataFrame,
        title: str,
        output_path: Path,
) -> None:

    if error_summary.empty:
        return

    row = error_summary.iloc[0]

    labels = [
        "Erfolgreich",
        "Fehler",
    ]

    values = [
        row["success_rate"],
        row["error_rate"],
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    sns.barplot(
        x=labels,
        y=values,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("Anteil")
    ax.set_xlabel("Ergebnis")
    ax.set_ylim(0, 1)

    for index, value in enumerate(values):
        ax.text(
            index,
            value + 0.02,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_error_categories(
        df: pd.DataFrame,
        column: str,
        title: str,
        output_path: Path,
) -> None:

    if df.empty or column not in df.columns:
        return

    plot_data = df.copy()

    if len(plot_data) > 20:
        plot_data = plot_data.head(20)

    fig_height = max(
        6,
        len(plot_data) * 0.45,
        )

    fig, ax = plt.subplots(
        figsize=(12, fig_height)
    )

    sns.barplot(
        data=plot_data,
        x="count",
        y=column,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Anzahl")
    ax.set_ylabel(column)

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def analyse_dataset(
        df: pd.DataFrame,
        dataset_name: str,
        output_dirs: dict,
) -> None:

    dataset_dir = (
            output_dirs["datasets"]
            / safe_name(dataset_name)
    )

    table_dir = dataset_dir / "tables"
    figure_dir = dataset_dir / "figures"

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics = calculate_dataset_statistics(
        df
    )

    save_table(
        statistics,
        table_dir / "overview.csv",
        )

    retrieval_summary = calculate_quality_summary(
        df,
        QUALITY_COLUMNS["retrieval"],
    )

    save_table(
        retrieval_summary,
        table_dir / "retrieval_quality.csv",
        )

    plot_quality(
        df,
        QUALITY_COLUMNS["retrieval"],
        f"{dataset_name} – Retrievalqualität",
        figure_dir / "retrieval_quality_distribution",
        )

    plot_quality_means(
        df,
        QUALITY_COLUMNS["retrieval"],
        f"{dataset_name} – Mittlere Retrievalqualität",
        figure_dir / "retrieval_quality_means",
        )

    answer_summary = calculate_quality_summary(
        df,
        QUALITY_COLUMNS["answer"],
    )

    save_table(
        answer_summary,
        table_dir / "answer_quality.csv",
        )

    plot_quality(
        df,
        QUALITY_COLUMNS["answer"],
        f"{dataset_name} – Antwortqualität",
        figure_dir / "answer_quality_distribution",
        )

    plot_quality_means(
        df,
        QUALITY_COLUMNS["answer"],
        f"{dataset_name} – Mittlere Antwortqualität",
        figure_dir / "answer_quality_means",
        )

    time_summary = calculate_summary(
        df,
        TIME_COLUMNS,
    )

    save_table(
        time_summary,
        table_dir / "times.csv",
        )

    plot_times(
        df,
        f"{dataset_name} – Laufzeitverteilung",
        figure_dir / "times_distribution",
        )

    plot_time_means(
        df,
        f"{dataset_name} – Mittlere Laufzeiten",
        figure_dir / "times_means",
        )

    hardware_summary = calculate_summary(
        df,
        HARDWARE_COLUMNS,
    )

    save_table(
        hardware_summary,
        table_dir / "hardware.csv",
        )

    plot_hardware(
        df,
        f"{dataset_name} – Hardware-Nutzung",
        figure_dir / "hardware_distribution",
        )

    plot_hardware_means(
        df,
        f"{dataset_name} – Mittlere Hardware-Nutzung",
        figure_dir / "hardware_means",
        )

    errors = calculate_error_analysis(df)

    save_table(
        errors["summary"],
        table_dir / "errors_summary.csv",
        )

    save_table(
        errors["phase"],
        table_dir / "errors_by_phase.csv",
        )

    save_table(
        errors["type"],
        table_dir / "errors_by_type.csv",
        )

    save_table(
        errors["description"],
        table_dir / "errors_by_description.csv",
        )

    plot_error_rate(
        errors["summary"],
        f"{dataset_name} – Fehlerquote",
        figure_dir / "error_rate",
        )

    plot_error_categories(
        errors["phase"],
        "error_phase",
        f"{dataset_name} – Fehler nach Phase",
        figure_dir / "errors_by_phase",
        )

    plot_error_categories(
        errors["type"],
        "error_type",
        f"{dataset_name} – Fehler nach Typ",
        figure_dir / "errors_by_type",
        )

    plot_error_categories(
        errors["description"],
        "error_message",
        f"{dataset_name} – Fehlerbeschreibungen",
        figure_dir / "errors_by_description",
        )


def combine_method_datasets(
        datasets: list[dict],
) -> dict[str, pd.DataFrame]:

    grouped = {}

    for item in datasets:
        method = normalize_method(
            item["method"]
        )

        input_df = load_input_file(
            item["input"]
        )

        hardware_df = load_hardware_file(
            item["hardware"]
        )

        input_df = convert_numeric_columns(
            input_df
        )

        hardware_df = convert_numeric_columns(
            hardware_df
        )

        merged = merge_input_and_hardware(
            input_df,
            hardware_df,
        )

        merged["method"] = method

        dataset_name = item.get(
            "name",
            Path(item["input"]).stem,
        )

        merged["dataset"] = dataset_name

        if method not in grouped:
            grouped[method] = []

        grouped[method].append(merged)

    return {
        method: pd.concat(
            frames,
            ignore_index=True,
        )
        for method, frames in grouped.items()
    }


def analyse_method(
        method: str,
        df: pd.DataFrame,
        output_dirs: dict,
) -> None:

    method_dir = (
            output_dirs["methods"]
            / safe_name(method)
    )

    table_dir = method_dir / "tables"
    figure_dir = method_dir / "figures"

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_table(
        df,
        table_dir / "combined_data.csv",
        )

    overview = calculate_dataset_statistics(
        df
    )

    save_table(
        overview,
        table_dir / "overview.csv",
        )

    retrieval_summary = calculate_quality_summary(
        df,
        QUALITY_COLUMNS["retrieval"],
    )

    save_table(
        retrieval_summary,
        table_dir / "retrieval_quality.csv",
        )

    plot_quality(
        df,
        QUALITY_COLUMNS["retrieval"],
        f"{method} – Retrievalqualität",
        figure_dir / "retrieval_quality_distribution",
        )

    plot_quality_means(
        df,
        QUALITY_COLUMNS["retrieval"],
        f"{method} – Mittlere Retrievalqualität",
        figure_dir / "retrieval_quality_means",
        )

    answer_summary = calculate_quality_summary(
        df,
        QUALITY_COLUMNS["answer"],
    )

    save_table(
        answer_summary,
        table_dir / "answer_quality.csv",
        )

    plot_quality(
        df,
        QUALITY_COLUMNS["answer"],
        f"{method} – Antwortqualität",
        figure_dir / "answer_quality_distribution",
        )

    plot_quality_means(
        df,
        QUALITY_COLUMNS["answer"],
        f"{method} – Mittlere Antwortqualität",
        figure_dir / "answer_quality_means",
        )

    time_summary = calculate_summary(
        df,
        TIME_COLUMNS,
    )

    save_table(
        time_summary,
        table_dir / "times.csv",
        )

    plot_times(
        df,
        f"{method} – Laufzeitverteilung",
        figure_dir / "times_distribution",
        )

    plot_time_means(
        df,
        f"{method} – Mittlere Laufzeiten",
        figure_dir / "times_means",
        )

    hardware_summary = calculate_summary(
        df,
        HARDWARE_COLUMNS,
    )

    save_table(
        hardware_summary,
        table_dir / "hardware.csv",
        )

    plot_hardware(
        df,
        f"{method} – Hardware-Nutzung",
        figure_dir / "hardware_distribution",
        )

    plot_hardware_means(
        df,
        f"{method} – Mittlere Hardware-Nutzung",
        figure_dir / "hardware_means",
        )

    errors = calculate_error_analysis(df)

    save_table(
        errors["summary"],
        table_dir / "errors_summary.csv",
        )

    save_table(
        errors["phase"],
        table_dir / "errors_by_phase.csv",
        )

    save_table(
        errors["type"],
        table_dir / "errors_by_type.csv",
        )

    save_table(
        errors["description"],
        table_dir / "errors_by_description.csv",
        )

    plot_error_rate(
        errors["summary"],
        f"{method} – Fehlerquote",
        figure_dir / "error_rate",
        )

    plot_error_categories(
        errors["phase"],
        "error_phase",
        f"{method} – Fehler nach Phase",
        figure_dir / "errors_by_phase",
        )

    plot_error_categories(
        errors["type"],
        "error_type",
        f"{method} – Fehler nach Typ",
        figure_dir / "errors_by_type",
        )

    plot_error_categories(
        errors["description"],
        "error_message",
        f"{method} – Fehlerbeschreibungen",
        figure_dir / "errors_by_description",
        )


def create_comparison_tables(
        method_data: dict[str, pd.DataFrame],
        output_dirs: dict,
) -> dict:

    comparison_dir = output_dirs["comparison"]
    table_dir = comparison_dir / "tables"

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for method, df in method_data.items():

        row = {
            "method": method,
            "questions": len(df),
        }

        for column in (
                QUALITY_COLUMNS["retrieval"]
                + QUALITY_COLUMNS["answer"]
                + TIME_COLUMNS
                + HARDWARE_COLUMNS
        ):
            if column in df.columns:
                row[f"{column}_mean"] = (
                    df[column].mean()
                )

        errors = calculate_error_analysis(df)

        if not errors["summary"].empty:
            error_row = errors["summary"].iloc[0]

            row["errors"] = error_row["errors"]
            row["error_rate"] = (
                error_row["error_rate"]
            )
            row["success_rate"] = (
                error_row["success_rate"]
            )

        rows.append(row)

    comparison = pd.DataFrame(rows)

    save_table(
        comparison,
        table_dir / "method_comparison.csv",
        )

    return comparison


def plot_method_quality_comparison(
        method_data: dict[str, pd.DataFrame],
        columns: list[str],
        title: str,
        output_path: Path,
) -> None:

    rows = []

    for method, df in method_data.items():
        for column in columns:
            if column in df.columns:
                rows.append(
                    {
                        "method": method,
                        "metric": column,
                        "value": df[column].mean(),
                    }
                )

    if not rows:
        return

    plot_data = pd.DataFrame(rows)

    metric_names = {
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "context_relevance": "Context Relevancy",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
    }

    plot_data["metric"] = (
        plot_data["metric"]
        .map(metric_names)
        .fillna(plot_data["metric"])
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    sns.barplot(
        data=plot_data,
        x="metric",
        y="value",
        hue="method",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Metrik")
    ax.set_ylabel("Mittlerer Score")
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_method_time_comparison(
        method_data: dict[str, pd.DataFrame],
        output_path: Path,
) -> None:

    rows = []

    names = {
        "retrieval_time": "Retrieval",
        "translation_time": "Translation",
        "generate_answer_time": "Generation",
        "total_time": "Gesamt",
    }

    for method, df in method_data.items():
        for column in TIME_COLUMNS:
            if column in df.columns:
                rows.append(
                    {
                        "method": method,
                        "metric": names.get(
                            column,
                            column,
                        ),
                        "value": df[column].mean(),
                    }
                )

    if not rows:
        return

    plot_data = pd.DataFrame(rows)

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    sns.barplot(
        data=plot_data,
        x="metric",
        y="value",
        hue="method",
        ax=ax,
    )

    ax.set_title(
        "Vergleich der mittleren Laufzeiten"
    )

    ax.set_xlabel("Messgröße")
    ax.set_ylabel("Mittlere Zeit")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_method_hardware_comparison(
        method_data: dict[str, pd.DataFrame],
        output_path: Path,
) -> None:

    rows = []

    names = {
        "avg_ram_mb": "Ø RAM",
        "peak_ram_mb": "Peak RAM",
        "avg_cpu_percent": "Ø CPU",
        "peak_cpu_percent": "Peak CPU",
    }

    for method, df in method_data.items():
        for column in HARDWARE_COLUMNS:
            if column in df.columns:
                rows.append(
                    {
                        "method": method,
                        "metric": names.get(
                            column,
                            column,
                        ),
                        "value": df[column].mean(),
                    }
                )

    if not rows:
        return

    plot_data = pd.DataFrame(rows)

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    sns.barplot(
        data=plot_data,
        x="metric",
        y="value",
        hue="method",
        ax=ax,
    )

    ax.set_title(
        "Vergleich der mittleren Hardware-Nutzung"
    )

    ax.set_xlabel("Messgröße")
    ax.set_ylabel("Mittlerer Wert")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_method_error_comparison(
        method_data: dict[str, pd.DataFrame],
        output_path: Path,
) -> None:

    rows = []

    for method, df in method_data.items():

        errors = calculate_error_analysis(
            df
        )

        if errors["summary"].empty:
            continue

        row = errors["summary"].iloc[0]

        rows.append(
            {
                "method": method,
                "error_rate": row["error_rate"],
                "success_rate": row["success_rate"],
            }
        )

    if not rows:
        return

    plot_data = pd.DataFrame(rows)

    plot_data = plot_data.melt(
        id_vars="method",
        value_vars=[
            "error_rate",
            "success_rate",
        ],
        var_name="metric",
        value_name="value",
    )

    plot_data["metric"] = plot_data[
        "metric"
    ].replace(
        {
            "error_rate": "Fehlerquote",
            "success_rate": "Erfolgsquote",
        }
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    sns.barplot(
        data=plot_data,
        x="method",
        y="value",
        hue="metric",
        ax=ax,
    )

    ax.set_title(
        "Vergleich von Fehler- und Erfolgsquoten"
    )

    ax.set_xlabel("Methode")
    ax.set_ylabel("Anteil")
    ax.set_ylim(0, 1)

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_method_error_categories(
        method_data: dict[str, pd.DataFrame],
        column: str,
        title: str,
        output_path: Path,
) -> None:

    rows = []

    for method, df in method_data.items():

        errors = calculate_error_analysis(
            df
        )

        if column == "error_phase":
            data = errors["phase"]

        elif column == "error_type":
            data = errors["type"]

        elif column == "error_message":
            data = errors["description"]

        else:
            continue

        if data.empty:
            continue

        for _, row in data.iterrows():
            rows.append(
                {
                    "method": method,
                    "category": row[column],
                    "count": row["count"],
                }
            )

    if not rows:
        return

    plot_data = pd.DataFrame(rows)

    top_categories = (
        plot_data
        .groupby("category")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    plot_data = plot_data[
        plot_data["category"].isin(
            top_categories
        )
    ]

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    sns.barplot(
        data=plot_data,
        x="count",
        y="category",
        hue="method",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Anzahl")
    ax.set_ylabel("Kategorie")

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def create_dataset_comparison(
        datasets: list[dict],
        output_dirs: dict,
) -> pd.DataFrame:

    rows = []

    for item in datasets:

        input_df = load_input_file(
            item["input"]
        )

        hardware_df = load_hardware_file(
            item["hardware"]
        )

        input_df = convert_numeric_columns(
            input_df
        )

        hardware_df = convert_numeric_columns(
            hardware_df
        )

        df = merge_input_and_hardware(
            input_df,
            hardware_df,
        )

        method = normalize_method(
            item["method"]
        )

        dataset_name = item.get(
            "name",
            Path(item["input"]).stem,
        )

        row = {
            "dataset": dataset_name,
            "method": method,
            "questions": len(df),
        }

        for column in (
                QUALITY_COLUMNS["retrieval"]
                + QUALITY_COLUMNS["answer"]
                + TIME_COLUMNS
                + HARDWARE_COLUMNS
        ):
            if column in df.columns:
                row[f"{column}_mean"] = (
                    df[column].mean()
                )

        errors = calculate_error_analysis(
            df
        )

        if not errors["summary"].empty:
            error_row = errors["summary"].iloc[0]

            row["errors"] = error_row["errors"]
            row["error_rate"] = (
                error_row["error_rate"]
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    save_table(
        result,
        output_dirs["comparison"]
        / "tables"
        / "dataset_comparison.csv",
        )

    return result


def plot_dataset_quality_comparison(
        dataset_comparison: pd.DataFrame,
        columns: list[str],
        title: str,
        output_path: Path,
) -> None:

    rows = []

    for _, row in dataset_comparison.iterrows():

        for column in columns:

            value_column = f"{column}_mean"

            if value_column not in dataset_comparison.columns:
                continue

            value = row[value_column]

            if pd.isna(value):
                continue

            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "metric": column,
                    "value": value,
                }
            )

    if not rows:
        return

    plot_data = pd.DataFrame(rows)

    names = {
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "context_relevance": "Context Relevancy",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
    }

    plot_data["metric"] = (
        plot_data["metric"]
        .map(names)
        .fillna(plot_data["metric"])
    )

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    sns.barplot(
        data=plot_data,
        x="dataset",
        y="value",
        hue="metric",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mittlerer Score")
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=35,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def plot_dataset_time_comparison(
        dataset_comparison: pd.DataFrame,
        output_path: Path,
) -> None:

    rows = []

    names = {
        "retrieval_time": "Retrieval",
        "translation_time": "Translation",
        "generate_answer_time": "Generation",
        "total_time": "Gesamt",
    }

    for _, row in dataset_comparison.iterrows():

        for column in TIME_COLUMNS:

            value_column = f"{column}_mean"

            if value_column not in dataset_comparison.columns:
                continue

            value = row[value_column]

            if pd.isna(value):
                continue

            rows.append(
                {
                    "dataset": row["dataset"],
                    "metric": names.get(
                        column,
                        column,
                    ),
                    "value": value,
                }
            )

    if not rows:
        return

    plot_data = pd.DataFrame(rows)

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    sns.barplot(
        data=plot_data,
        x="dataset",
        y="value",
        hue="metric",
        ax=ax,
    )

    ax.set_title(
        "Vergleich der Laufzeiten aller Datasets"
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mittlere Zeit")

    plt.xticks(
        rotation=35,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_path,
    )


def trigger_analyse_and_visualise(
        dataset: list[dict],
        output_dir: str = "analysis_results",
) -> None:

    print(
        "Start Analyse und Visualisation"
    )

    if not dataset:
        raise ValueError(
            "Die Dataset-Liste darf nicht leer sein."
        )

    results_dir = resolve_project_path(ANALYSIS_RESULTS_DIRECTORY)

    output_dirs = create_directories(
        Path(results_dir) / output_dir
    )

    all_method_data = combine_method_datasets(
        dataset
    )

    dataset_comparison = create_dataset_comparison(
        dataset,
        output_dirs,
    )

    for item in dataset:

        input_df = load_input_file(
            item["input"]
        )

        hardware_df = load_hardware_file(
            item["hardware"]
        )

        input_df = convert_numeric_columns(
            input_df
        )

        hardware_df = convert_numeric_columns(
            hardware_df
        )

        merged_df = merge_input_and_hardware(
            input_df,
            hardware_df,
        )

        method = normalize_method(
            item["method"]
        )

        dataset_name = item.get(
            "name",
            Path(item["input"]).stem,
        )

        dataset_output_dir = (
                output_dirs["datasets"]
                / safe_name(dataset_name)
        )

        save_phase_hardware_analysis(
            hardware_df,
            method,
            dataset_output_dir / "tables",
            dataset_output_dir / "figures",
            dataset_name,
            )

        merged_df["method"] = method
        merged_df["dataset"] = dataset_name

        analyse_dataset(
            merged_df,
            dataset_name,
            output_dirs,
        )

    for method, df in all_method_data.items():

        analyse_method(
            method,
            df,
            output_dirs,
        )

        method_items = [
            item
            for item in dataset
            if normalize_method(
                item["method"]
            ) == method
        ]

        method_phase_rows = []

        for method_item in method_items:

            method_hardware_df = load_hardware_file(
                method_item["hardware"]
            )

            method_hardware_df = convert_numeric_columns(
                method_hardware_df
            )

            method_phase_df = calculate_phase_hardware_analysis(
                method_hardware_df,
                method=method,
            )

            if not method_phase_df.empty:
                method_phase_rows.append(
                    method_phase_df
                )

        if method_phase_rows:

            method_phase_df = pd.concat(
                method_phase_rows,
                ignore_index=True,
            )

            method_phase_df = (
                method_phase_df
                .groupby(
                    ["method", "phase"],
                    as_index=False,
                )[HARDWARE_COLUMNS]
                .mean()
            )

            method_dir = (
                    output_dirs["methods"]
                    / safe_name(method)
            )

            save_table(
                method_phase_df,
                method_dir
                / "tables"
                / "hardware_by_phase.csv",
                )

            plot_phase_hardware_ram(
                method_phase_df,
                f"{method} – Ø RAM nach Phase",
                method_dir
                / "figures"
                / "hardware_avg_ram_by_phase",
                )

            plot_phase_hardware_peak_ram(
                method_phase_df,
                f"{method} – Peak RAM nach Phase",
                method_dir
                / "figures"
                / "hardware_peak_ram_by_phase",
                )

            plot_phase_hardware_cpu(
                method_phase_df,
                f"{method} – Ø CPU nach Phase",
                method_dir
                / "figures"
                / "hardware_avg_cpu_by_phase",
                )

            plot_phase_hardware_peak_cpu(
                method_phase_df,
                f"{method} – Peak CPU nach Phase",
                method_dir
                / "figures"
                / "hardware_peak_cpu_by_phase",
                )

    phase_hardware_by_method = (
        combine_phase_hardware_by_method(
            dataset
        )
    )

    if not phase_hardware_by_method.empty:

        save_table(
            phase_hardware_by_method,
            output_dirs["comparison"]
            / "tables"
            / "hardware_by_method_and_phase.csv",
            )

    comparison_dir = (
            output_dirs["comparison"]
            / "figures"
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_method_phase_hardware_comparison(
        phase_hardware_by_method,
        comparison_dir,
    )

    plot_method_quality_comparison(
        all_method_data,
        QUALITY_COLUMNS["retrieval"],
        "Vergleich der Retrievalqualität",
        comparison_dir
        / "methods_retrieval_quality",
        )

    plot_method_quality_comparison(
        all_method_data,
        QUALITY_COLUMNS["answer"],
        "Vergleich der Antwortqualität",
        comparison_dir
        / "methods_answer_quality",
        )

    plot_method_time_comparison(
        all_method_data,
        comparison_dir
        / "methods_times",
        )

    plot_method_hardware_comparison(
        all_method_data,
        comparison_dir
        / "methods_hardware",
        )

    plot_method_error_comparison(
        all_method_data,
        comparison_dir
        / "methods_error_rate",
        )

    plot_method_error_categories(
        all_method_data,
        "error_phase",
        "Vergleich der Fehler nach Phase",
        comparison_dir
        / "methods_errors_by_phase",
        )

    plot_method_error_categories(
        all_method_data,
        "error_type",
        "Vergleich der Fehler nach Typ",
        comparison_dir
        / "methods_errors_by_type",
        )

    plot_method_error_categories(
        all_method_data,
        "error_message",
        "Vergleich der Fehlerbeschreibungen",
        comparison_dir
        / "methods_errors_by_description",
        )

    plot_dataset_quality_comparison(
        dataset_comparison,
        QUALITY_COLUMNS["retrieval"],
        "Retrievalqualität der einzelnen Datasets",
        comparison_dir
        / "datasets_retrieval_quality",
        )

    plot_dataset_quality_comparison(
        dataset_comparison,
        QUALITY_COLUMNS["answer"],
        "Antwortqualität der einzelnen Datasets",
        comparison_dir
        / "datasets_answer_quality",
        )

    plot_dataset_time_comparison(
        dataset_comparison,
        comparison_dir
        / "datasets_times",
        )

    print(
        "Analyse und Visualisation finished"
    )

    print(
        "Ergebnisse gespeichert unter: "
        f"{output_dirs['root'].resolve()}"
    )


if __name__ == "__main__":

    result_dir = "results"
    hardware_dir = "hardware"

    dataset = [
        {
            "method": "trag",
            "name": "tRAG_fr_context_de",
            "input": (
                f"{result_dir}/tRAG/"
                "questions_fr_context_de.csv"
            ),
            "hardware": (
                f"{hardware_dir}/tRAG/"
                "questions_fr_context_de.csv"
            ),
        },
        {
            "method": "trag",
            "name": "tRAG_fr_context_en",
            "input": (
                f"{result_dir}/tRAG/"
                "questions_fr_context_en.csv"
            ),
            "hardware": (
                f"{hardware_dir}/tRAG/"
                "questions_fr_context_en.csv"
            ),
        },
        {
            "method": "monorag",
            "name": "MonoRAG_fr_context_fr",
            "input": (
                f"{result_dir}/monoRAG/"
                "questions_fr_context_fr.csv"
            ),
            "hardware": (
                f"{hardware_dir}/monoRAG/"
                "questions_fr_context_fr.csv"
            ),
        },
        {
            "method": "multirag",
            "name": "MultiRAG_fr_context_de_en_fr",
            "input": (
                f"{result_dir}/multiRAG/"
                "questions_fr_context_de_en_fr.csv"
            ),
            "hardware": (
                f"{hardware_dir}/multiRAG/"
                "questions_fr_context_de_en_fr.csv"
            ),
        },
        {
            "method": "crossrag",
            "name": "CrossRAG_fr_context_de",
            "input": (
                f"{result_dir}/crossRAG/"
                "questions_fr_context_de.csv"
            ),
            "hardware": (
                f"{hardware_dir}/crossRAG/"
                "questions_fr_context_de.csv"
            ),
        },
    ]

    trigger_analyse_and_visualise(
        dataset=dataset,
        output_dir="analysis_results",
    )