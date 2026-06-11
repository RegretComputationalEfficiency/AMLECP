from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "2014_combined_pickup_datetime_taxi_with_directionality.csv"
OUTPUT_CSV = BASE_DIR / "2014_filtered_combined_pickup_datetime_taxi_with_directionary.csv"
PICKUP_TIME_COL = "pickup_time"
CHUNKSIZE = 500_000


def filter_valid_pickup_time_and_save(input_filename=INPUT_CSV, output_filename=OUTPUT_CSV):
    """
    Check that pickup_time can be parsed as a datetime, drop invalid rows,
    and save the valid rows to a new CSV.
    """
    input_path = Path(input_filename)
    output_path = Path(output_filename)

    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    min_pickup_time = None
    max_pickup_time = None
    write_header = True

    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=CHUNKSIZE), start=1):
        if PICKUP_TIME_COL not in chunk.columns:
            raise ValueError(
                f"Column '{PICKUP_TIME_COL}' not found in {input_path}. "
                f"Columns: {chunk.columns.tolist()}"
            )

        total_rows += len(chunk)
        parsed_pickup_time = pd.to_datetime(chunk[PICKUP_TIME_COL], errors="coerce")
        valid_mask = parsed_pickup_time.notna()
        invalid_count = int((~valid_mask).sum())
        invalid_rows += invalid_count

        valid_chunk = chunk.loc[valid_mask].copy()
        valid_chunk[PICKUP_TIME_COL] = parsed_pickup_time.loc[valid_mask].dt.strftime("%Y-%m-%d %H:%M:%S")

        if not valid_chunk.empty:
            chunk_min = parsed_pickup_time.loc[valid_mask].min()
            chunk_max = parsed_pickup_time.loc[valid_mask].max()
            min_pickup_time = chunk_min if min_pickup_time is None else min(min_pickup_time, chunk_min)
            max_pickup_time = chunk_max if max_pickup_time is None else max(max_pickup_time, chunk_max)

        valid_chunk.to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False
        valid_rows += len(valid_chunk)

        print(
            f"chunk {chunk_index}: read {len(chunk):,}, "
            f"valid pickup_time {len(valid_chunk):,}, invalid {invalid_count:,}"
        )

    print(f"Done. Read rows: {total_rows:,}")
    print(f"Valid pickup_time rows: {valid_rows:,}")
    print(f"Invalid pickup_time rows dropped: {invalid_rows:,}")
    print(f"pickup_time range: {min_pickup_time} to {max_pickup_time}")
    print(f"Saved filtered data to: {output_path}")


if __name__ == "__main__":
    filter_valid_pickup_time_and_save()
