"""Utilities for association-rule analysis on social media activity data.

The functions in this module are intentionally notebook-friendly: each step is
small, inspectable, and can be adjusted while deciding which columns to keep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


TARGET_COLUMN = "self_reported_happiness"


DEMOGRAPHIC_COLUMNS = [
    "age",
    "gender",
    "country",
    "urban_rural",
    "income_level",
    "employment_status",
    "education_level",
    "relationship_status",
    "has_children",
]

LIFESTYLE_COLUMNS = [
    "exercise_hours_per_week",
    "sleep_hours_per_night",
    "diet_quality",
    "smoking",
    "perceived_stress_score",
    "self_reported_happiness",
    "body_mass_index",
    "daily_steps_count",
    "weekly_work_hours",
]

DIGITAL_ACTIVITY_COLUMNS = [
    "sessions_per_day",
    "average_session_length_minutes",
    "posts_created_per_week",
    "likes_given_per_day",
    "comments_written_per_day",
    "dms_sent_per_week",
    "followers_count",
    "following_count",
]

APP_TIME_COLUMNS = [
    "time_on_feed_per_day",
    "time_on_explore_per_day",
    "time_on_messages_per_day",
    "time_on_reels_per_day",
]

PREFERENCE_COLUMNS = [
    "content_type_preference",
    "preferred_content_theme",
    "privacy_setting_level",
]


def load_dataset(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV dataset from a local path."""
    return pd.read_csv(path, **read_csv_kwargs)


def summarize_columns(df: pd.DataFrame, sample_values: int = 5) -> pd.DataFrame:
    """Return a compact summary to support manual column selection."""
    rows = []
    for column in df.columns:
        non_null_values = df[column].dropna().unique()[:sample_values]
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": round(float(df[column].isna().mean() * 100), 3),
                "unique_count": int(df[column].nunique(dropna=True)),
                "example_values": list(non_null_values),
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages by column."""
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean() * 100,
        }
    )
    return report.sort_values("missing_count", ascending=False)


def report_duplicates(df: pd.DataFrame) -> dict[str, int | float]:
    """Return duplicate-row counts and percentages."""
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = float(df.duplicated().mean() * 100)
    return {"duplicate_count": duplicate_count, "duplicate_pct": duplicate_pct}


def split_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split dataframe columns into numeric and categorical groups."""
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    return {"numeric": numeric_columns, "categorical": categorical_columns}


def select_columns(df: pd.DataFrame, selected_columns: Sequence[str]) -> pd.DataFrame:
    """Return a dataframe with only the manually selected columns."""
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Selected columns not found in dataframe: {missing_columns}")
    return df.loc[:, list(selected_columns)].copy()


def clean_selected_data(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "unknown",
    unknown_label: str = "Unknown",
) -> pd.DataFrame:
    """Clean missing values without making irreversible column choices.

    Numeric strategies: "median", "mean", "zero", or "drop".
    Categorical strategies: "unknown", "mode", or "drop".
    """
    cleaned = df.copy()
    column_types = split_column_types(cleaned)

    if numeric_strategy == "drop" or categorical_strategy == "drop":
        subset = []
        if numeric_strategy == "drop":
            subset.extend(column_types["numeric"])
        if categorical_strategy == "drop":
            subset.extend(column_types["categorical"])
        cleaned = cleaned.dropna(subset=subset)

    for column in column_types["numeric"]:
        if not cleaned[column].isna().any():
            continue
        if numeric_strategy == "median":
            fill_value = cleaned[column].median()
        elif numeric_strategy == "mean":
            fill_value = cleaned[column].mean()
        elif numeric_strategy == "zero":
            fill_value = 0
        elif numeric_strategy == "drop":
            continue
        else:
            raise ValueError(f"Unsupported numeric_strategy: {numeric_strategy}")
        cleaned[column] = cleaned[column].fillna(fill_value)

    for column in column_types["categorical"]:
        if not cleaned[column].isna().any():
            continue
        if categorical_strategy == "unknown":
            fill_value = unknown_label
        elif categorical_strategy == "mode":
            mode_values = cleaned[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else unknown_label
        elif categorical_strategy == "drop":
            continue
        else:
            raise ValueError(f"Unsupported categorical_strategy: {categorical_strategy}")
        cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned.reset_index(drop=True)


def _safe_interval_label(column: str, interval: pd.Interval | object) -> str:
    """Convert qcut interval labels into readable item names."""
    if isinstance(interval, pd.Interval):
        left = f"{interval.left:.3g}"
        right = f"{interval.right:.3g}"
        return f"{column}=[{left}, {right}]"
    return f"{column}={interval}"


def discretize_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    max_bins: int = 10,
) -> pd.DataFrame:
    """Discretize numeric columns with quantiles, limited to max_bins bins.

    Columns with a single unique value are converted to a constant categorical
    item. qcut may create fewer than max_bins bins when duplicate edges exist.
    """
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")

    transformed = df.copy()
    numeric_columns = split_column_types(transformed)["numeric"]
    target_columns = list(columns) if columns is not None else numeric_columns

    for column in target_columns:
        if column not in transformed.columns:
            raise ValueError(f"Column not found: {column}")
        if column not in numeric_columns:
            continue

        unique_count = transformed[column].nunique(dropna=True)
        if unique_count <= 1:
            transformed[column] = transformed[column].map(lambda value: f"{column}={value}")
            continue

        bins = min(max_bins, unique_count)
        try:
            discretized = pd.qcut(transformed[column], q=bins, duplicates="drop")
        except ValueError:
            discretized = pd.cut(transformed[column], bins=bins, duplicates="drop")

        transformed[column] = discretized.map(lambda interval: _safe_interval_label(column, interval))

    return transformed


def create_binary_matrix(df: pd.DataFrame, sparse: bool = False) -> pd.DataFrame:
    """Create a one-hot encoded binary matrix for association-rule mining."""
    encoded = pd.get_dummies(df.astype("string"), prefix_sep="=", sparse=sparse)
    encoded.columns = [str(column) for column in encoded.columns]
    return encoded.astype(bool)


def mine_frequent_itemsets(
    binary_df: pd.DataFrame,
    min_support: float = 0.01,
    use_colnames: bool = True,
) -> pd.DataFrame:
    """Mine frequent itemsets with FP-Growth."""
    if binary_df.empty:
        raise ValueError("binary_df is empty")
    return fpgrowth(binary_df, min_support=min_support, use_colnames=use_colnames)


def generate_rules(
    itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.4,
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets."""
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    if rules.empty:
        return rules
    return rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def filter_rules_by_item(rules: pd.DataFrame, item_name: str) -> pd.DataFrame:
    """Filter rules where the item text appears in antecedents or consequents."""
    if rules.empty:
        return rules.copy()

    item_name = item_name.lower()

    def contains_item(items: frozenset[str]) -> bool:
        return any(item_name in str(item).lower() for item in items)

    mask = rules["antecedents"].map(contains_item) | rules["consequents"].map(contains_item)
    return rules.loc[mask].reset_index(drop=True)


def filter_happiness_rules(
    rules: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    consequent_only: bool = False,
) -> pd.DataFrame:
    """Filter association rules linked to self-reported happiness."""
    if rules.empty:
        return rules.copy()

    target_column = target_column.lower()

    def contains_target(items: frozenset[str]) -> bool:
        return any(target_column in str(item).lower() for item in items)

    if consequent_only:
        mask = rules["consequents"].map(contains_target)
    else:
        mask = rules["antecedents"].map(contains_target) | rules["consequents"].map(contains_target)
    return rules.loc[mask].reset_index(drop=True)


def rules_to_readable(rules: pd.DataFrame) -> pd.DataFrame:
    """Convert frozenset columns into readable strings for display/export."""
    if rules.empty:
        return rules.copy()

    readable = rules.copy()
    readable["antecedents"] = readable["antecedents"].map(lambda values: " AND ".join(sorted(map(str, values))))
    readable["consequents"] = readable["consequents"].map(lambda values: " AND ".join(sorted(map(str, values))))
    return readable


def sample_dataframe(
    df: pd.DataFrame,
    use_sample: bool = True,
    sample_size: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Optionally sample large datasets for faster notebook iterations."""
    if not use_sample or len(df) <= sample_size:
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def validate_binary_matrix(binary_df: pd.DataFrame) -> dict[str, object]:
    """Return simple validation metadata for the encoded matrix."""
    dtypes = binary_df.dtypes.astype(str).value_counts().to_dict()
    return {
        "rows": int(binary_df.shape[0]),
        "columns": int(binary_df.shape[1]),
        "dtype_counts": dtypes,
        "has_missing_values": bool(binary_df.isna().any().any()),
    }


def suggest_existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    """Return candidate columns that actually exist in the dataframe."""
    return [column for column in candidates if column in df.columns]


def default_analysis_columns(df: pd.DataFrame) -> list[str]:
    """Suggest a starting column set based on the project documentation."""
    candidates = (
        DEMOGRAPHIC_COLUMNS
        + LIFESTYLE_COLUMNS
        + DIGITAL_ACTIVITY_COLUMNS
        + APP_TIME_COLUMNS
        + PREFERENCE_COLUMNS
    )
    return suggest_existing_columns(df, candidates)


def reduce_rare_categories(
    df: pd.DataFrame,
    categorical_columns: Iterable[str] | None = None,
    min_frequency: float = 0.005,
    other_label: str = "Other",
) -> pd.DataFrame:
    """Group rare categorical values to control binary matrix width."""
    if not 0 <= min_frequency <= 1:
        raise ValueError("min_frequency must be between 0 and 1")

    reduced = df.copy()
    columns = list(categorical_columns) if categorical_columns is not None else split_column_types(reduced)["categorical"]

    for column in columns:
        if column not in reduced.columns:
            raise ValueError(f"Column not found: {column}")
        frequencies = reduced[column].value_counts(normalize=True, dropna=False)
        rare_values = frequencies[frequencies < min_frequency].index
        reduced[column] = np.where(reduced[column].isin(rare_values), other_label, reduced[column])

    return reduced
