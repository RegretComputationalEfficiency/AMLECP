import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
tqdm.pandas(desc="Generating synthetic samples")

# Same directory as this script (Simulator/); produced by load_data_with_map.py (2013_4_…).
DEFAULT_PROCESSED_DATA_PATH = str(
    Path(__file__).resolve().parent / "2013_4_Trained_nyc_taxi_data_with_directionality.csv"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "beta_sensitivity_outputs_2013_4"
DEFAULT_RANDOM_SEED = 2025


PAPER4_EXPERIMENTS = [
    {"base_scenario": "pair_1p5", "multipliers": [1.5], "noise_rate": 0.05},
    {"base_scenario": "pair_1p5_2p0", "multipliers": [1.5, 2.0], "noise_rate": 0.05},
    {"base_scenario": "pair_1p5_2p0", "multipliers": [1.5, 2.0], "noise_rate": 0.10},
    {"base_scenario": "pair_1p5_2p0", "multipliers": [1.5, 2.0], "noise_rate": 0.20},
]

SCENARIO_LIBRARY = {
    "pair_1p5": {"multipliers": [1.5]},
    "pair_1p5_2p0": {"multipliers": [1.5, 2.0]},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train beta_simulator under multiple reject-generation settings.")
    parser.add_argument("--processed-data-path", default=DEFAULT_PROCESSED_DATA_PATH, help="Path to processed training CSV.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder to store per-scenario outputs.",
    )
    parser.add_argument(
        "--scenario-set",
        choices=["paper4", "full"],
        default="paper4",
        help="'paper4': 1.2/2.0/2.5×noise_00 + 2.0×noise_10 (4 runs). "
        "'full': all SCENARIO_LIBRARY entries × --noise-rates.",
    )
    parser.add_argument(
        "--noise-rates",
        default="0.00,0.05,0.10",
        help="Used only with --scenario-set full: comma-separated reject-label flip rates (0->1).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed.")
    return parser.parse_args()


def get_contextual_features(df):
    if "fare_amount" not in df.columns:
        raise ValueError("Missing required column 'fare_amount'.")

    contextual_features = [col for col in df.columns if col != "fare_amount"]
    cols_to_remove = {
        "distance_bin",
        "hour_bin",
        "day_type_bin",
        "minute_bin",
        "second_bin",
        "segment",
        "typical_fare_segment",
    }
    return [c for c in contextual_features if c not in cols_to_remove]


def add_segmentation_columns(df):
    if "distance_mi" in df.columns:
        base_distance_col = "distance_mi"
    elif "distance_km" in df.columns:
        base_distance_col = "distance_km"
    else:
        base_distance_col = None

    minute_col = "pickup_minute" if "pickup_minute" in df.columns else "pickup_minute_of_hour"
    second_col = "pickup_second" if "pickup_second" in df.columns else "pickup_second_of_minute"

    segment_cols = []
    if base_distance_col:
        df["distance_bin"] = pd.cut(
            df[base_distance_col],
            bins=[0, 2, 5, 10, np.inf],
            labels=["dist_short", "dist_medium", "dist_long", "dist_very_long"],
            right=False,
        )
        segment_cols.append("distance_bin")

    if "pickup_hour_of_day" in df.columns:
        df["hour_bin"] = pd.cut(
            df["pickup_hour_of_day"],
            bins=[-1, 5, 11, 17, 23],
            labels=["hour_night", "hour_morning", "hour_afternoon", "hour_evening"],
            right=True,
            include_lowest=True,
        )
        segment_cols.append("hour_bin")

    if "pickup_day_of_week" in df.columns:
        df["day_type_bin"] = pd.cut(
            df["pickup_day_of_week"],
            bins=[-1, 4, 6],
            labels=["weekday", "weekend"],
            right=True,
            include_lowest=True,
        )
        segment_cols.append("day_type_bin")

    if minute_col in df.columns:
        df["minute_bin"] = pd.cut(
            df[minute_col],
            bins=[-1, 14, 29, 44, 59],
            labels=["min_0_14", "min_15_29", "min_30_44", "min_45_59"],
            right=True,
            include_lowest=True,
        )
        segment_cols.append("minute_bin")

    if second_col in df.columns:
        df["second_bin"] = pd.cut(
            df[second_col],
            bins=[-1, 14, 29, 44, 59],
            labels=["sec_0_14", "sec_15_29", "sec_30_44", "sec_45_59"],
            right=True,
            include_lowest=True,
        )
        segment_cols.append("second_bin")

    if segment_cols:
        df["segment"] = df[segment_cols].astype(str).agg("_".join, axis=1)
    else:
        df["segment"] = "global_segment"

    return df


def attach_typical_fare(df):
    typical_prices = df.groupby("segment")["fare_amount"].median().rename("typical_fare_segment")
    out = df.merge(typical_prices, on="segment", how="left")
    global_median = out["fare_amount"].median()
    if pd.isna(global_median):
        global_median = 15.0
    out["typical_fare_segment"] = out["typical_fare_segment"].fillna(global_median)
    return out


def build_augmented_data(df, contextual_features, multipliers, noise_rate, rng):
    augmented = []

    df_positive = df.copy()
    df_positive["price_offered"] = df_positive["fare_amount"]
    df_positive["accepted"] = 1
    pos_cols = [c for c in contextual_features if c in df_positive.columns] + ["price_offered", "accepted"]
    augmented.extend(df_positive[pos_cols].to_dict(orient="records"))

    reject_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building rejects {multipliers}"):
        typical_fare = row.get("typical_fare_segment")
        if pd.isna(typical_fare):
            continue

        base_features = {feat: row[feat] for feat in contextual_features if feat in row and pd.notna(row[feat])}
        for multiplier in multipliers:
            synthetic_price = min(max(float(typical_fare) * float(multiplier), 0.01), 499.99)
            reject_sample = base_features.copy()
            reject_sample["price_offered"] = synthetic_price
            reject_sample["accepted"] = 0
            reject_rows.append(reject_sample)

    if noise_rate > 0 and reject_rows:
        flip_count = int(round(noise_rate * len(reject_rows)))
        if flip_count > 0:
            flip_idx = rng.choice(len(reject_rows), size=flip_count, replace=False)
            for idx in flip_idx:
                reject_rows[idx]["accepted"] = 1

    augmented.extend(reject_rows)
    df_augmented = pd.DataFrame(augmented)
    return df_augmented, len(reject_rows)


def train_beta_model(df_augmented, contextual_features):
    interaction_features = []
    for feature in contextual_features:
        interaction_feature_name = f"price_x_{feature}"
        if (
            feature in df_augmented.columns
            and pd.api.types.is_numeric_dtype(df_augmented["price_offered"])
            and pd.api.types.is_numeric_dtype(df_augmented[feature])
        ):
            df_augmented[interaction_feature_name] = df_augmented["price_offered"] * df_augmented[feature]
            interaction_features.append(interaction_feature_name)

    features_for_model = [f for f in contextual_features + interaction_features if f in df_augmented.columns]
    if not features_for_model:
        raise ValueError("No valid features available for model training.")

    train_df = df_augmented[features_for_model + ["accepted"]].copy()
    train_df.dropna(inplace=True)
    if train_df.empty or train_df["accepted"].nunique() <= 1:
        raise ValueError("Training data is empty or contains only one class.")

    X = train_df[features_for_model].copy()
    y = train_df["accepted"]

    numeric_cols = []
    for col in features_for_model:
        if X[col].nunique() > 2 and pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
    numeric_cols = sorted(set(numeric_cols))

    scaler = None
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        random_state=42,
        class_weight="balanced",
        max_iter=1000,
    )
    model.fit(X, y)

    beta_df = pd.DataFrame(
        {
            "feature": ["intercept"] + X.columns.tolist(),
            "coefficient": [model.intercept_[0]] + model.coef_[0].tolist(),
        }
    )
    return beta_df, scaler, numeric_cols, float(y.mean()), len(train_df)


def save_scenario_outputs(output_dir, df_augmented, beta_df, scaler, numeric_cols, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_augmented.to_csv(output_dir / "augmented_training_data.csv", index=False)
    beta_df.to_csv(output_dir / "beta_simulator_parameters.csv", index=False)
    if scaler is not None and numeric_cols:
        joblib.dump(scaler, output_dir / "beta_simulator_scaler.joblib")
        with open(output_dir / "beta_simulator_scaled_features_list.json", "w", encoding="utf-8") as f:
            json.dump(numeric_cols, f)
    else:
        with open(output_dir / "beta_simulator_scaled_features_list.json", "w", encoding="utf-8") as f:
            json.dump([], f)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.scenario_set == "paper4":
        experiment_runs = list(PAPER4_EXPERIMENTS)
    else:
        noise_rates = [float(x.strip()) for x in args.noise_rates.split(",") if x.strip()]
        experiment_runs = [
            {
                "base_scenario": name,
                "multipliers": cfg["multipliers"],
                "noise_rate": nr,
            }
            for name, cfg in SCENARIO_LIBRARY.items()
            for nr in noise_rates
        ]

    print(f"Loading processed data from: {args.processed_data_path}")
    df = pd.read_csv(args.processed_data_path)
    if df.empty:
        raise ValueError("Processed dataset is empty.")

    contextual_features = get_contextual_features(df)
    df = add_segmentation_columns(df)
    df = attach_typical_fare(df)

    summary_rows = []
    for run in experiment_runs:
        scenario_name = run["base_scenario"]
        multipliers = run["multipliers"]
        noise_rate = float(run["noise_rate"])
        noise_tag = f"noise_{int(round(noise_rate * 100)):02d}"
        full_name = f"{scenario_name}__{noise_tag}"
        print(f"\n=== Training scenario: {full_name} ===")

        df_augmented, reject_count = build_augmented_data(
            df=df,
            contextual_features=contextual_features,
            multipliers=multipliers,
            noise_rate=noise_rate,
            rng=rng,
        )
        beta_df, scaler, numeric_cols, acceptance_rate, train_size = train_beta_model(
            df_augmented=df_augmented,
            contextual_features=contextual_features,
        )

        metadata = {
            "scenario_name": full_name,
            "base_scenario": scenario_name,
            "reject_multipliers": multipliers,
            "noise_rate_reject_flip_0_to_1": noise_rate,
            "seed": args.seed,
            "augmented_rows": int(len(df_augmented)),
            "reject_rows_generated": int(reject_count),
            "training_rows_after_dropna": int(train_size),
            "acceptance_ratio_after_augmentation": acceptance_rate,
        }
        save_scenario_outputs(
            output_dir=output_root / full_name,
            df_augmented=df_augmented,
            beta_df=beta_df,
            scaler=scaler,
            numeric_cols=numeric_cols,
            metadata=metadata,
        )
        summary_rows.append(metadata)
        print(f"Saved to: {output_root / full_name}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "scenario_summary.csv", index=False)
    print(f"\nCompleted {len(summary_df)} scenarios. Summary: {output_root / 'scenario_summary.csv'}")


if __name__ == "__main__":
    main()
