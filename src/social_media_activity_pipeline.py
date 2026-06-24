"""Pipeline utilities for association-rule experiments on social media data."""

from __future__ import annotations

import re
import unicodedata
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


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


def normalize_name(value: object) -> str:
    """Normalize column names and categorical values to stable snake_case text."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def load_dataset(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV dataset from a local path."""
    return pd.read_csv(path, **read_csv_kwargs)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with normalized and unique column names."""
    normalized = df.copy()
    used_names: dict[str, int] = {}
    new_columns = []

    for column in normalized.columns:
        base_name = normalize_name(column)
        count = used_names.get(base_name, 0)
        used_names[base_name] = count + 1
        new_columns.append(base_name if count == 0 else f"{base_name}_{count + 1}")

    normalized.columns = new_columns
    return normalized


def normalize_categorical_values(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Normalize categorical text values without modifying numeric columns."""
    normalized = df.copy()
    selected_columns = list(columns) if columns is not None else split_column_types(normalized)["categorical"]

    for column in selected_columns:
        if column not in normalized.columns:
            raise ValueError(f"Column not found: {column}")
        normalized[column] = normalized[column].map(normalize_name).astype("string")

    return normalized


def summarize_columns(df: pd.DataFrame, sample_values: int = 5) -> pd.DataFrame:
    """Return a compact summary to support manual column selection."""
    rows = []
    row_count = max(len(df), 1)

    for column in df.columns:
        non_null_values = df[column].dropna().unique()[:sample_values]
        unique_count = int(df[column].nunique(dropna=True))
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": round(float(df[column].isna().mean() * 100), 3),
                "unique_count": unique_count,
                "unique_ratio": round(unique_count / row_count, 6),
                "example_values": list(non_null_values),
            }
        )

    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def analyze_column_cardinality(df: pd.DataFrame, sample_values: int = 5) -> pd.DataFrame:
    """Return column metadata plus a suggested action for cleaning."""
    summary = summarize_columns(df, sample_values=sample_values)
    actions = []
    reasons = []

    for row in summary.itertuples(index=False):
        column = row.column
        missing_pct = row.missing_pct
        unique_count = row.unique_count
        unique_ratio = row.unique_ratio
        dtype = row.dtype

        if column == TARGET_COLUMN:
            action = "keep"
            reason = "target column"
        elif missing_pct >= 70:
            action = "drop_candidate"
            reason = "too many missing values"
        elif unique_count <= 1:
            action = "drop_candidate"
            reason = "constant or nearly empty column"
        elif unique_ratio >= 0.95 and unique_count > 100:
            action = "drop_candidate"
            reason = "likely identifier or high-cardinality free text"
        elif "object" in dtype.lower() and unique_count > 100:
            action = "review"
            reason = "high-cardinality categorical column"
        else:
            action = "keep"
            reason = "usable for association rules"

        actions.append(action)
        reasons.append(reason)

    summary["suggested_action"] = actions
    summary["reason"] = reasons
    return summary


def suggest_columns_to_drop(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    max_missing_pct: float = 70,
    max_unique_ratio: float = 0.95,
    max_categorical_unique: int = 100,
) -> list[str]:
    """Suggest uninformative columns to drop before binary encoding."""
    row_count = max(len(df), 1)
    columns_to_drop = []

    for column in df.columns:
        if column == target_column:
            continue

        missing_pct = float(df[column].isna().mean() * 100)
        unique_count = int(df[column].nunique(dropna=True))
        unique_ratio = unique_count / row_count
        is_categorical = column in split_column_types(df)["categorical"]

        if missing_pct >= max_missing_pct:
            columns_to_drop.append(column)
        elif unique_count <= 1:
            columns_to_drop.append(column)
        elif unique_ratio >= max_unique_ratio and unique_count > 100:
            columns_to_drop.append(column)
        elif is_categorical and unique_count > max_categorical_unique:
            columns_to_drop.append(column)

    return columns_to_drop


def drop_columns(df: pd.DataFrame, columns_to_drop: Sequence[str]) -> pd.DataFrame:
    """Drop selected columns while ignoring names that are not present."""
    existing_columns = [column for column in columns_to_drop if column in df.columns]
    return df.drop(columns=existing_columns).copy()


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
    duplicate_mask = df.duplicated()
    return {
        "duplicate_count": int(duplicate_mask.sum()),
        "duplicate_pct": float(duplicate_mask.mean() * 100),
    }


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
    unknown_label: str = "unknown",
) -> pd.DataFrame:
    """Clean missing values without making irreversible column choices."""
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


def sample_dataframe(
    df: pd.DataFrame,
    use_sample: bool = True,
    sample_size: int = 50_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Optionally sample large datasets for fast functional tests."""
    if not use_sample or len(df) <= sample_size:
        return df.copy().reset_index(drop=True)
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _safe_interval_label(column: str, interval: pd.Interval | object) -> str:
    """Convert interval labels into readable item names."""
    if isinstance(interval, pd.Interval):
        left = f"{interval.left:.3g}"
        right = f"{interval.right:.3g}"
        return f"{column}__{left}_to_{right}"
    return f"{column}__{normalize_name(interval)}"


def discretize_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    max_bins: int = 10,
) -> pd.DataFrame:
    """Discretize numeric columns with quantiles, limited to max_bins bins."""
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
            transformed[column] = transformed[column].map(lambda value: f"{column}__{value}").astype("string")
            continue

        bins = min(max_bins, unique_count)
        try:
            discretized = pd.qcut(transformed[column], q=bins, duplicates="drop")
        except ValueError:
            discretized = pd.cut(transformed[column], bins=bins, duplicates="drop")

        transformed[column] = discretized.map(lambda interval: _safe_interval_label(column, interval)).astype("string")

    return transformed


def reduce_rare_categories(
    df: pd.DataFrame,
    categorical_columns: Iterable[str] | None = None,
    min_frequency: float = 0.005,
    other_label: str = "other",
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


def create_binary_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dense boolean one-hot matrix compatible with mlxtend."""
    encoded = pd.get_dummies(df.astype("string"), prefix_sep="=")
    encoded.columns = [str(column) for column in encoded.columns]
    return encoded.astype(bool)


def validate_binary_matrix(binary_df: pd.DataFrame) -> dict[str, object]:
    """Return simple validation metadata for the encoded matrix."""
    dtypes = binary_df.dtypes.astype(str).value_counts().to_dict()
    return {
        "rows": int(binary_df.shape[0]),
        "columns": int(binary_df.shape[1]),
        "dtype_counts": dtypes,
        "has_missing_values": bool(binary_df.isna().any().any()),
    }


def binary_matrix_to_transactions(binary_df: pd.DataFrame) -> list[list[str]]:
    """Convert a boolean matrix into transaction lists for Eclat."""
    transactions: list[list[str]] = []
    columns = binary_df.columns.to_numpy()

    for row in binary_df.to_numpy(dtype=bool):
        transactions.append(columns[row].tolist())

    return transactions


def transactions_to_dataframe(transactions: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Convert variable-length transactions to the dataframe shape used by pyECLAT."""
    max_length = max((len(transaction) for transaction in transactions), default=0)
    padded_rows = [list(transaction) + [np.nan] * (max_length - len(transaction)) for transaction in transactions]
    return pd.DataFrame(padded_rows)


def run_apriori_algorithm(binary_df: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
    """Run Apriori and return frequent itemsets in mlxtend format."""
    return apriori(binary_df, min_support=min_support, use_colnames=True)


def run_fpgrowth_algorithm(binary_df: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
    """Run FP-Growth and return frequent itemsets in mlxtend format."""
    return fpgrowth(binary_df, min_support=min_support, use_colnames=True)


def run_eclat_algorithm(
    binary_df: pd.DataFrame,
    min_support: float = 0.02,
    min_combination: int = 1,
    max_combination: int = 3,
) -> pd.DataFrame:
    """Run Eclat with pyECLAT and return itemsets in mlxtend-compatible format."""
    try:
        from pyECLAT import ECLAT
    except ImportError as exc:
        raise ImportError("Install pyECLAT in Colab with: !pip install -q pyECLAT") from exc

    transactions = binary_matrix_to_transactions(binary_df)
    transactions_df = transactions_to_dataframe(transactions)
    try:
        eclat_instance = ECLAT(data=transactions_df, verbose=False)
    except TypeError:
        eclat_instance = ECLAT(data=transactions_df)

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        try:
            _, supports = eclat_instance.fit(
                min_support=min_support,
                min_combination=min_combination,
                max_combination=max_combination,
                separator=" & ",
                verbose=False,
            )
        except TypeError:
            _, supports = eclat_instance.fit(
                min_support=min_support,
                min_combination=min_combination,
                max_combination=max_combination,
            )

    rows = []
    for itemset_text, support in supports.items():
        if isinstance(itemset_text, str):
            items = [item.strip() for item in itemset_text.split(" & ") if item.strip()]
        else:
            items = list(itemset_text)
        rows.append({"support": float(support), "itemsets": frozenset(items)})

    return pd.DataFrame(rows, columns=["support", "itemsets"])


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


def run_algorithm_experiment(
    binary_df: pd.DataFrame,
    algorithm: str,
    min_support: float = 0.02,
    min_confidence: float = 0.4,
    min_lift: float = 1.0,
    target_column: str = TARGET_COLUMN,
    eclat_max_combination: int = 3,
) -> dict[str, pd.DataFrame | str]:
    """Run one algorithm and return itemsets, rules, and target-related rules."""
    algorithm_key = normalize_name(algorithm)

    if algorithm_key == "apriori":
        itemsets = run_apriori_algorithm(binary_df, min_support=min_support)
    elif algorithm_key in {"fp_growth", "fpgrowth"}:
        itemsets = run_fpgrowth_algorithm(binary_df, min_support=min_support)
        algorithm_key = "fp_growth"
    elif algorithm_key == "eclat":
        itemsets = run_eclat_algorithm(
            binary_df,
            min_support=min_support,
            max_combination=eclat_max_combination,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    rules = generate_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules[rules["lift"] >= min_lift].reset_index(drop=True)

    happiness_rules = filter_happiness_rules(rules, target_column=target_column)

    return {
        "algorithm": algorithm_key,
        "itemsets": itemsets,
        "rules": rules,
        "happiness_rules": happiness_rules,
    }


def run_all_algorithms(
    binary_df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.4,
    min_lift: float = 1.0,
    target_column: str = TARGET_COLUMN,
    eclat_max_combination: int = 3,
) -> dict[str, dict[str, pd.DataFrame | str]]:
    """Run Apriori, FP-Growth, and Eclat with the same thresholds."""
    results = {}
    for algorithm in ["apriori", "fp_growth", "eclat"]:
        results[algorithm] = run_algorithm_experiment(
            binary_df=binary_df,
            algorithm=algorithm,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            target_column=target_column,
            eclat_max_combination=eclat_max_combination,
        )
    return results


def summarize_algorithm_results(results: dict[str, dict[str, pd.DataFrame | str]]) -> pd.DataFrame:
    """Compare algorithms by rule quality and usefulness, not by runtime."""
    rows = []

    for algorithm, result in results.items():
        itemsets = result["itemsets"]
        rules = result["rules"]
        happiness_rules = result["happiness_rules"]

        if not isinstance(itemsets, pd.DataFrame) or not isinstance(rules, pd.DataFrame):
            raise TypeError("Algorithm result must contain dataframe itemsets and rules")
        if not isinstance(happiness_rules, pd.DataFrame):
            raise TypeError("Algorithm result must contain dataframe happiness_rules")

        quality_source = happiness_rules if not happiness_rules.empty else rules
        rows.append(
            {
                "algorithm": algorithm,
                "frequent_itemsets_count": int(len(itemsets)),
                "rules_count": int(len(rules)),
                "happiness_rules_count": int(len(happiness_rules)),
                "avg_support": _metric_mean(quality_source, "support"),
                "avg_confidence": _metric_mean(quality_source, "confidence"),
                "avg_lift": _metric_mean(quality_source, "lift"),
                "max_lift": _metric_max(quality_source, "lift"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["happiness_rules_count", "avg_lift", "avg_confidence", "rules_count"],
        ascending=[False, False, False, False],
    )


def choose_best_algorithm(summary: pd.DataFrame) -> str:
    """Choose the top algorithm from the quality-focused comparison table."""
    if summary.empty:
        return "no_algorithm_selected"
    return str(summary.iloc[0]["algorithm"])


def _metric_mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(df[column].mean())


def _metric_max(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(df[column].max())


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
