"""Settings of build_data.py: data splits, cleaning rules and features of the HVFHV case study (2026-11-17).

High Volume For-Hire Vehicle trips (Uber HV0003, Lyft HV0005) of the NYC TLC. A request is priced at its
request time, and the recorded upfront fare base_passenger_fare is the historical price.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
RAW = DATA / "raw"                            # fhvhv_tripdata_<YYYY-MM>.parquet from the TLC trip record page
PROC = DATA / "processed"
ZONES_DBF = DATA / "zones" / "taxi_zones" / "taxi_zones.dbf"
WEATHER_CSV = DATA / "weather" / "daily_weather_2025_2026.csv"   # Open-Meteo historical weather API, NYC

# ---- data splits (request time, New York local time)
FIT_MONTHS = ["2025-10", "2025-11", "2025-12"] + [f"2026-{m:02d}" for m in range(1, 8)]   # 10 months
VAL_MONTH, TEST_MONTH = "2026-08", "2026-09"
VAL_DAYS = ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"]   # Mondays of the validation month
TEST_DAYS = ["2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28"]  # Mondays of the test month
FIT_SAMPLE = 6_000_000                        # candidate trips for the simulator fit (run.py prepare draws 2,000,000)
SEED_DATA = 20260901                          # random offsets of requests recorded in the same second

# ---- cleaning, identical for all splits
LICENSES = ("HV0003", "HV0005")               # Uber, Lyft
PRICE_LO, PRICE_HI = 5.0, 150.0               # feasible prices [l, u] (about the 1% and 99.5% quantiles)
DIST_LO, DIST_HI = 0.1, 50.0                  # miles

# ---- features (20 contextual features, kappa = 1 + 20 + 20 = 41). Column 0 is the platform (Lyft = 1);
# the other 19 are defined as in the 2014 taxi study. Column 16 is is_long_daylight (daylight >= 12 h).
FEATURES = [
    "is_lyft", "distance_mi",
    "pickup_in_manhattan", "dropoff_in_manhattan",
    "pickup_near_airport", "dropoff_near_airport",
    "traffic_congestion",
    "pickup_in_business_district", "dropoff_in_business_district",
    "is_morning_commute", "is_evening_commute",
    "is_biz_to_airport", "is_airport_to_biz",
    "had_rain", "had_snow_fall", "is_extreme_temp", "is_long_daylight",
    "is_weekend", "is_night", "is_cross_borough",
]
BUSINESS_ZONES = {"Midtown Center", "Financial District North",
                  "Times Sq/Theatre District", "Upper East Side South", "Midtown East"}
AIRPORT_ZONE_IDS = {1, 132, 138}              # Newark, JFK, LaGuardia
