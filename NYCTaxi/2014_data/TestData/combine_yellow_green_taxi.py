import argparse
import csv
import heapq
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
SIMULATOR_DIR = PROJECT_DIR / "Simulator"

GREEN_PARQUET = BASE_DIR / "green_tripdata_2014-12.parquet"
YELLOW_PARQUET = BASE_DIR / "yellow_tripdata_2014-12.parquet"
OUTPUT_CSV = BASE_DIR / "2014_combined_pickup_datetime_taxi_with_directionality.csv"
UNSORTED_OUTPUT_CSV = BASE_DIR / "2014_combined_pickup_datetime_taxi_with_directionality.unsorted.tmp.csv"
SORTED_OUTPUT_CSV = BASE_DIR / "2014_combined_pickup_datetime_taxi_with_directionality.sorted.tmp.csv"
# Not shipped in the repo — download from Open-Meteo (see Simulator/README.md).
WEATHER_CSV = SIMULATOR_DIR / "daily_weather_data_2009_2015.csv"

CHUNKSIZE = 500_000

# Feature columns stay in exactly the same order as Simulator/2013_Trained_nyc_taxi_data_with_directionality.csv.
# The original pickup timestamp is kept as the final extra column.
OUTPUT_COLUMNS = [
    "fare_amount",
    "passenger_count",
    "distance_mi",
    "pickup_in_manhattan",
    "dropoff_in_manhattan",
    "pickup_near_airport",
    "dropoff_near_airport",
    "traffic_congestion",
    "pickup_is_business_district",
    "dropoff_is_business_district",
    "is_pickup_residential_proxy",
    "is_dropoff_residential_proxy",
    "is_morning_commute",
    "is_evening_commute",
    "is_biz_to_airport",
    "is_airport_to_biz",
    "had_rain",
    "had_snow_fall",
    "is_extreme_temp",
    "is_long_daylight",
    "is_long_sunshine",
    "pickup_time",
]

COMMON_COLUMNS = [
    "pickup_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "PULocationID",
    "DOLocationID",
]

BUSINESS_ZONE_NAMES = {
    "Midtown Center",
    "Financial District North",
    "Times Sq/Theatre District",
    "Upper East Side South",
    "Midtown East",
}
AIRPORT_LOCATION_IDS = {1, 132, 138}


def _resolve_taxi_zones_path():
    # Download taxi_zones.zip from NYC TLC Taxi Zone Maps (see Simulator/README.md).
    for candidate in (
        SIMULATOR_DIR / "taxi_zones" / "taxi_zones.shp",
        SIMULATOR_DIR / "taxi_zones.shp",
        SIMULATOR_DIR / "taxi_zones.zip",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Taxi zone file not found. Expected one of: "
        "Simulator/taxi_zones/taxi_zones.shp, Simulator/taxi_zones.shp, Simulator/taxi_zones.zip"
    )


def load_zone_features():
    zones = gpd.read_file(_resolve_taxi_zones_path())[
        ["LocationID", "borough", "zone", "geometry"]
    ].copy()
    zones["LocationID"] = pd.to_numeric(zones["LocationID"], errors="coerce").astype("Int64")

    projected = zones.to_crs(epsg=2263)
    centroids = projected.geometry.centroid
    centroid_latlon = gpd.GeoSeries(centroids, crs=projected.crs).to_crs(epsg=4326)

    zones["zone_latitude"] = centroid_latlon.y
    zones["zone_longitude"] = centroid_latlon.x
    zones["in_manhattan"] = (zones["borough"] == "Manhattan").astype(int)
    zones["near_airport"] = (
        zones["LocationID"].isin(AIRPORT_LOCATION_IDS)
        | zones["zone"].str.contains("Airport|Newark", case=False, na=False)
    ).astype(int)
    zones["is_business_district"] = zones["zone"].isin(BUSINESS_ZONE_NAMES).astype(int)

    return zones.set_index("LocationID")[
        [
            "zone_latitude",
            "zone_longitude",
            "in_manhattan",
            "near_airport",
            "is_business_district",
        ]
    ]


def load_weather_features():
    if not WEATHER_CSV.exists():
        raise FileNotFoundError(
            f"Weather CSV not found: {WEATHER_CSV}\n"
            "Download daily NYC weather from the Open-Meteo Historical Weather API "
            "(see Simulator/README.md) and save as Simulator/daily_weather_data_2009_2015.csv."
        )

    weather = pd.read_csv(WEATHER_CSV)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()

    numeric_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "rain_sum",
        "snowfall_sum",
        "daylight_duration",
        "sunshine_duration",
    ]
    for col in numeric_cols:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather["had_rain"] = (weather.get("rain_sum", 0) > 0).astype(int)
    weather["had_snow_fall"] = (weather.get("snowfall_sum", 0) > 0).astype(int)
    weather["is_extreme_temp"] = (
        (weather.get("temperature_2m_max", 0) > 32)
        | (weather.get("temperature_2m_min", 0) < 0)
    ).astype(int)

    weather_2014 = weather[
        (weather["date"] >= pd.Timestamp("2014-01-01"))
        & (weather["date"] <= pd.Timestamp("2014-12-31"))
    ]
    daylight_median = weather_2014.get("daylight_duration", pd.Series(dtype=float)).median()
    sunshine_median = weather_2014.get("sunshine_duration", pd.Series(dtype=float)).median()

    weather["is_long_daylight"] = 0
    weather["is_long_sunshine"] = 0
    if pd.notna(daylight_median) and "daylight_duration" in weather.columns:
        weather["is_long_daylight"] = (weather["daylight_duration"] >= daylight_median).astype(int)
    if pd.notna(sunshine_median) and "sunshine_duration" in weather.columns:
        if sunshine_median > 0:
            weather["is_long_sunshine"] = (weather["sunshine_duration"] >= sunshine_median).astype(int)
        else:
            weather["is_long_sunshine"] = (weather["sunshine_duration"] > 0).astype(int)

    return weather[
        [
            "date",
            "had_rain",
            "had_snow_fall",
            "is_extreme_temp",
            "is_long_daylight",
            "is_long_sunshine",
        ]
    ]


def standardize_chunk(chunk, pickup_col):
    chunk = chunk.rename(columns={pickup_col: "pickup_datetime"}).copy()
    return chunk[COMMON_COLUMNS]


def add_zone_columns(df, zone_features):
    pickup_features = zone_features.add_prefix("pickup_")
    dropoff_features = zone_features.add_prefix("dropoff_")

    df = df.join(pickup_features, on="PULocationID")
    df = df.join(dropoff_features, on="DOLocationID")

    rename_map = {
        "pickup_in_manhattan": "pickup_in_manhattan",
        "dropoff_in_manhattan": "dropoff_in_manhattan",
        "pickup_near_airport": "pickup_near_airport",
        "dropoff_near_airport": "dropoff_near_airport",
        "pickup_is_business_district": "pickup_is_business_district",
        "dropoff_is_business_district": "dropoff_is_business_district",
    }
    df = df.rename(columns=rename_map)

    binary_cols = [
        "pickup_in_manhattan",
        "dropoff_in_manhattan",
        "pickup_near_airport",
        "dropoff_near_airport",
        "pickup_is_business_district",
        "dropoff_is_business_district",
    ]
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

    return df


def process_chunk(chunk, pickup_col, zone_features, weather_features):
    df = standardize_chunk(chunk, pickup_col)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
    df["distance_mi"] = pd.to_numeric(df["trip_distance"], errors="coerce")
    df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce").astype("Int64")
    df["DOLocationID"] = pd.to_numeric(df["DOLocationID"], errors="coerce").astype("Int64")

    df = df.dropna(
        subset=[
            "pickup_datetime",
            "fare_amount",
            "passenger_count",
            "distance_mi",
            "PULocationID",
            "DOLocationID",
        ]
    )
    df = df[
        (df["fare_amount"] > 0)
        & (df["fare_amount"] < 500)
        & (df["passenger_count"].between(1, 6))
        & (df["distance_mi"] > 0.01)
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["pickup_time"] = df["pickup_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["passenger_count"] = df["passenger_count"].astype(int)
    df = add_zone_columns(df, zone_features)

    pickup_hour = df["pickup_datetime"].dt.hour
    pickup_day_of_week = df["pickup_datetime"].dt.dayofweek
    df["traffic_congestion"] = (
        pickup_hour.between(8, 18) & pickup_day_of_week.between(0, 4)
    ).astype(int)

    df["is_pickup_residential_proxy"] = (
        (df["pickup_is_business_district"] == 0) & (df["pickup_near_airport"] == 0)
    ).astype(int)
    df["is_dropoff_residential_proxy"] = (
        (df["dropoff_is_business_district"] == 0) & (df["dropoff_near_airport"] == 0)
    ).astype(int)
    df["is_morning_commute"] = (
        pickup_hour.between(6, 9)
        & (df["is_pickup_residential_proxy"] == 1)
        & (df["dropoff_is_business_district"] == 1)
    ).astype(int)
    df["is_evening_commute"] = (
        pickup_hour.between(17, 19)
        & (df["pickup_is_business_district"] == 1)
        & (df["is_dropoff_residential_proxy"] == 1)
    ).astype(int)
    df["is_biz_to_airport"] = (
        (df["pickup_is_business_district"] == 1) & (df["dropoff_near_airport"] == 1)
    ).astype(int)
    df["is_airport_to_biz"] = (
        (df["pickup_near_airport"] == 1) & (df["dropoff_is_business_district"] == 1)
    ).astype(int)

    if weather_features is not None:
        df["date"] = df["pickup_datetime"].dt.normalize()
        df = df.merge(weather_features, on="date", how="left")
        df = df.drop(columns=["date"])

    weather_cols = [
        "had_rain",
        "had_snow_fall",
        "is_extreme_temp",
        "is_long_daylight",
        "is_long_sunshine",
    ]
    for col in weather_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    return df[OUTPUT_COLUMNS]


def _iter_input_chunks(input_path: Path, pickup_col: str, usecols: list[str]):
    if input_path.suffix == ".parquet":
        if pq is None:
            raise ImportError("Reading parquet requires pyarrow. Install with: pip install pyarrow")
        parquet_file = pq.ParquetFile(input_path)
        for batch in parquet_file.iter_batches(batch_size=CHUNKSIZE, columns=usecols):
            yield batch.to_pandas()
        return

    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=CHUNKSIZE):
        yield chunk


def append_processed_file(input_path, pickup_col, output_csv, zone_features, weather_features, write_header):
    total_rows = 0
    usecols = [pickup_col, "passenger_count", "trip_distance", "fare_amount", "PULocationID", "DOLocationID"]

    for chunk_index, chunk in enumerate(_iter_input_chunks(input_path, pickup_col, usecols), start=1):
        processed = process_chunk(chunk, pickup_col, zone_features, weather_features)
        processed.to_csv(output_csv, mode="a", header=write_header, index=False)
        write_header = False
        total_rows += len(processed)
        print(f"{input_path.name}: chunk {chunk_index} processed, kept {len(processed):,} rows")

    return total_rows, write_header


def sort_csv_by_pickup_time(input_csv, output_csv):
    sorted_chunk_paths = []

    with tempfile.TemporaryDirectory(prefix="pickup_time_sort_", dir=BASE_DIR) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for chunk_index, chunk in enumerate(pd.read_csv(input_csv, chunksize=CHUNKSIZE), start=1):
            chunk = chunk.sort_values("pickup_time", kind="mergesort")
            chunk_path = temp_dir / f"sorted_chunk_{chunk_index:05d}.csv"
            chunk.to_csv(chunk_path, index=False)
            sorted_chunk_paths.append(chunk_path)
            print(f"Sorted temporary chunk {chunk_index}: {len(chunk):,} rows")

        files = [path.open("r", newline="") for path in sorted_chunk_paths]
        try:
            readers = [csv.DictReader(file) for file in files]
            merged_rows = heapq.merge(*readers, key=lambda row: row["pickup_time"])

            with output_csv.open("w", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(merged_rows)
        finally:
            for file in files:
                file.close()


def validate_output_columns(csv_path=OUTPUT_CSV):
    output_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if output_columns != OUTPUT_COLUMNS:
        raise ValueError(
            "Output columns do not match the simulator feature schema plus pickup_time.\n"
            f"Expected: {OUTPUT_COLUMNS}\n"
            f"Actual:   {output_columns}"
        )
    print("Column validation passed: feature columns match beta_simulator training schema, with pickup_time appended.")


def validate_pickup_time_sorted(csv_path=OUTPUT_CSV):
    previous_pickup_time = None
    total_rows = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(csv_path, usecols=["pickup_time"], chunksize=CHUNKSIZE),
        start=1,
    ):
        pickup_times = chunk["pickup_time"]
        if not pickup_times.is_monotonic_increasing:
            raise ValueError(f"pickup_time is not sorted within chunk {chunk_index}.")
        first_pickup_time = pickup_times.iloc[0]
        last_pickup_time = pickup_times.iloc[-1]
        if previous_pickup_time is not None and first_pickup_time < previous_pickup_time:
            raise ValueError(
                "pickup_time is not globally sorted: "
                f"{first_pickup_time} appears after {previous_pickup_time}."
            )
        previous_pickup_time = last_pickup_time
        total_rows += len(chunk)

    print(f"Sort validation passed: pickup_time is ascending across {total_rows:,} rows.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge NYC yellow+green December 2014 TLC trips, engineer features, sort by pickup_time."
    )
    parser.add_argument("--green", type=Path, default=GREEN_PARQUET, help="Green taxi parquet/csv path.")
    parser.add_argument("--yellow", type=Path, default=YELLOW_PARQUET, help="Yellow taxi parquet/csv path.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Sorted combined output CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_csv = args.output
    unsorted_csv = output_csv.with_suffix(output_csv.suffix + ".unsorted.tmp.csv")
    sorted_csv = output_csv.with_suffix(output_csv.suffix + ".sorted.tmp.csv")

    for temp_csv in [unsorted_csv, sorted_csv]:
        temp_csv.unlink(missing_ok=True)

    print("Loading taxi zone and weather feature lookups...")
    zone_features = load_zone_features()
    weather_features = load_weather_features()

    print(f"Writing unsorted combined data to temporary file: {unsorted_csv}")
    write_header = True
    green_rows, write_header = append_processed_file(
        args.green, "lpep_pickup_datetime", unsorted_csv, zone_features, weather_features, write_header
    )
    yellow_rows, write_header = append_processed_file(
        args.yellow, "tpep_pickup_datetime", unsorted_csv, zone_features, weather_features, write_header
    )

    validate_output_columns(unsorted_csv)
    print(f"Sorting combined data by pickup_time ascending into: {sorted_csv}")
    sort_csv_by_pickup_time(unsorted_csv, sorted_csv)
    sorted_csv.replace(output_csv)
    unsorted_csv.unlink(missing_ok=True)

    validate_output_columns(output_csv)
    validate_pickup_time_sorted(output_csv)
    print(f"Done. Green rows kept: {green_rows:,}; yellow rows kept: {yellow_rows:,}; total: {green_rows + yellow_rows:,}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
