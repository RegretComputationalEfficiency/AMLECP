"""Settings of build_data.py: data splits, cleaning rules and features of the Section 5.2 NYC taxi case study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"                   # TLC parquet files (download, see ../README.md)
PROC = ROOT / "data" / "processed"            # written by build_data.py and make_cache.py
ZONES_DBF = ROOT.parent / "Simulator" / "taxi_zones" / "taxi_zones.dbf"
WEATHER_CSV = ROOT.parent / "Simulator" / "daily_weather_data_2009_2015.csv"

# ---- data splits (pickup time, New York local time, end exclusive) ----
FIT_MONTHS = ["2013-12"] + [f"2014-{m:02d}" for m in range(1, 12)]
FIT_SAMPLE = 6_000_000                        # candidate trips for the simulator fit (run.py prepare draws 2,000,000)
VAL_WINDOW = ("2014-11-17", "2014-11-24")     # validation week (hyperparameter selection)
SEED_DATA = 20141201                          # random offsets of trips recorded in the same second

# ---- cleaning, identical for all splits ----
PRICE_LO, PRICE_HI = 2.5, 70.0                # feasible prices [l, u]
DIST_LO, DIST_HI = 0.1, 30.0                  # miles

# ---- features (20 contextual features, kappa = 1 + 20 + 20 = 41) ----
# run.py replaces is_long_sunshine (column 16) by is_long_daylight, daylight of at least 12 hours on the trip's date.
FEATURES = [
    "passenger_count", "distance_mi",
    "pickup_in_manhattan", "dropoff_in_manhattan",
    "pickup_near_airport", "dropoff_near_airport",
    "traffic_congestion",
    "pickup_in_business_district", "dropoff_in_business_district",
    "is_morning_commute", "is_evening_commute",
    "is_biz_to_airport", "is_airport_to_biz",
    "had_rain", "had_snow_fall", "is_extreme_temp", "is_long_sunshine",
    "is_weekend", "is_night", "is_cross_borough",
]
BUSINESS_ZONES = {"Midtown Center", "Financial District North",
                  "Times Sq/Theatre District", "Upper East Side South", "Midtown East"}
AIRPORT_ZONE_IDS = {1, 132, 138}              # Newark, JFK, LaGuardia
