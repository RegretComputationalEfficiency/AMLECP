import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import osmnx as ox

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# Rows per chunk when reading; filter pickup_datetime to [T1, T2] inside each chunk
# so the full table never has to be loaded before time filtering.
READ_CSV_CHUNKSIZE = 500_000

# Time window applied while reading; only trips in this range are kept.
T1 = pd.to_datetime("2013-12-01 00:00:00", utc=True)
T2 = pd.to_datetime("2014-11-30 23:59:59", utc=True)

# Kaggle NYC train.csv: pickup_datetime format "2010-01-05 16:52:16 UTC" (space + UTC suffix).
PICKUP_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

# --- 0. Data Loading & Initial Inspection ---
print("--- 0. Loading Data & Initial Inspection ---")

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "train.csv"
RAW_ZIP = BASE_DIR / "train.csv.zip"

if RAW_CSV.exists():
    input_path = RAW_CSV
elif RAW_ZIP.exists():
    input_path = RAW_ZIP
else:
    raise FileNotFoundError(f"Cannot find raw data: {RAW_CSV} or {RAW_ZIP}")

print(f"Loading raw data from: {input_path}")
print(
    f"Chunked read (chunksize={READ_CSV_CHUNKSIZE}), keeping rows with "
    f"{T1} <= pickup_datetime <= {T2}"
)

filtered_chunks = []
reader = pd.read_csv(input_path, chunksize=READ_CSV_CHUNKSIZE, low_memory=False)
for chunk_idx, chunk in enumerate(reader):
    if "pickup_datetime" not in chunk.columns:
        raise ValueError(
            f"Column 'pickup_datetime' not found in {input_path}. Columns: {chunk.columns.tolist()}"
        )
    chunk["pickup_datetime"] = pd.to_datetime(
        chunk["pickup_datetime"],
        format=PICKUP_DATETIME_FORMAT,
        utc=True,
        errors="coerce",
    )
    chunk = chunk.dropna(subset=["pickup_datetime"])
    chunk = chunk[(chunk["pickup_datetime"] >= T1) & (chunk["pickup_datetime"] <= T2)]
    if not chunk.empty:
        filtered_chunks.append(chunk)
    if (chunk_idx + 1) % 20 == 0:
        n_kept = sum(len(c) for c in filtered_chunks)
        print(f"  ... raw chunks scanned: {chunk_idx + 1}, rows kept in window so far: {n_kept}")

df = (
    pd.concat(filtered_chunks, ignore_index=True)
    if filtered_chunks
    else pd.DataFrame()
)
print(f"Fare data loaded (time-filtered while reading). Shape: {df.shape}")

if "pickup_datetime" in df.columns and not df.empty:
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"],
            format=PICKUP_DATETIME_FORMAT,
            utc=True,
            errors="coerce",
        )
    elif df["pickup_datetime"].dt.tz is None:
        df["pickup_datetime"] = df["pickup_datetime"].dt.tz_localize("UTC")
    else:
        df["pickup_datetime"] = df["pickup_datetime"].dt.tz_convert("UTC")
    print(f"pickup_datetime dtype: {df['pickup_datetime'].dtype}")
    print("\n--- Sorting DataFrame by pickup_datetime ---")
    df.sort_values(by="pickup_datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("DataFrame sorted by 'pickup_datetime'.")
    print("*" * 40)
else:
    print("Warning: 'pickup_datetime' column missing or DataFrame empty after time-filtered load.")
    print("*" * 40)

# --- 0.5. Time window (already applied during chunked read) ---
print("\n--- 0.5. Time window ---")
print(f"All rows already restricted to pickup_datetime in [{T1}, {T2}] during load.")
print(f"Current shape: {df.shape}")
print("*" * 40)

# --- User Specific Requirement Filters (Fare, Passenger, Coordinates) ---
print("\n--- Applying User Specific Filters: Fare, Passenger, Coordinates ---")
if 'fare_amount' in df.columns and not df.empty:
    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce'); df.dropna(subset=['fare_amount'], inplace=True)
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)].copy()
if 'passenger_count' in df.columns and not df.empty:
    df['passenger_count'] = pd.to_numeric(df['passenger_count'], errors='coerce'); df.dropna(subset=['passenger_count'], inplace=True)
    df['passenger_count'] = df['passenger_count'].astype(float).astype(int)
    df = df[df['passenger_count'].between(1, 6)].copy()
if not df.empty and all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']):
    df = df[(df['pickup_latitude'].between(40.0, 42.0)) & (df['dropoff_latitude'].between(40.0, 42.0)) & \
            (df['pickup_longitude'].between(-74.0, -72.0)) & (df['dropoff_longitude'].between(-74.0, -72.0))].copy()
print(f"Shape after specific filters (fare, passenger, coordinates): {df.shape}")
print("*" * 40)


# --- 0.7 Load and Merge Weather Data (NEW SECTION) ---
print("\n--- 0.7 Loading and Merging Weather Data (daily_weather_data_2009_2015.csv) ---")
# Not shipped in the repo — download from Open-Meteo (see Simulator/README.md).
WEATHER_DATA_PATH = BASE_DIR / "daily_weather_data_2009_2015.csv"
df_weather = None
try:
    df_weather = pd.read_csv(WEATHER_DATA_PATH)
    print(f"Successfully loaded weather data '{WEATHER_DATA_PATH}'. Shape: {df_weather.shape}")

    df_weather['date'] = pd.to_datetime(df_weather['date'], format='%Y-%m-%d') 
    
    weather_numeric_cols = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 
                            'rain_sum', 'snowfall_sum', 'daylight_duration', 'sunshine_duration']
    for col in weather_numeric_cols:
        if col in df_weather.columns:
            df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')
        else:
            print(f"Warning: Expected weather column '{col}' not found in weather data.")

    if 'rain_sum' in df_weather.columns:
        df_weather['had_rain'] = (df_weather['rain_sum'] > 0).astype(int)
    else: df_weather['had_rain'] = 0
        
    if 'snowfall_sum' in df_weather.columns:
        df_weather['had_snow_fall'] = (df_weather['snowfall_sum'] > 0).astype(int)
    else: df_weather['had_snow_fall'] = 0
    
    if 'temperature_2m_max' in df_weather.columns and 'temperature_2m_min' in df_weather.columns:
        # Assuming Celsius: >32C is hot, <0C is cold
        df_weather['is_extreme_temp'] = ((df_weather['temperature_2m_max'] > 32) | (df_weather['temperature_2m_min'] < 0)).astype(int)
    else: df_weather['is_extreme_temp'] = 0

    # --- NEW: Categorize daylight_duration and sunshine_duration ---
    # First, filter weather data for the relevant period (2010-2011) to calculate representative medians
    weather_T1 = pd.to_datetime('2013-01-01')
    weather_T2 = pd.to_datetime('2014-11-30')
    df_weather_period = df_weather[(df_weather['date'] >= weather_T1) & (df_weather['date'] <= weather_T2)].copy()

    if not df_weather_period.empty:
        if 'daylight_duration' in df_weather_period.columns:
            median_daylight = df_weather_period['daylight_duration'].median()
            df_weather['is_long_daylight'] = (df_weather['daylight_duration'] >= median_daylight).astype(int)
            print(f"Categorized 'daylight_duration' based on median: {median_daylight:.2f} (1 for long, 0 for short/night)")
        else:
            df_weather['is_long_daylight'] = 0
            print("Warning: 'daylight_duration' not found in weather data for categorization.")

        if 'sunshine_duration' in df_weather_period.columns:
            median_sunshine = df_weather_period['sunshine_duration'].median()
            # Handle cases where sunshine might be 0 for many days (e.g. cloudy), median could be 0
            # If median is 0, 'long_sunshine' means any sunshine > 0.
            if median_sunshine > 0:
                df_weather['is_long_sunshine'] = (df_weather['sunshine_duration'] >= median_sunshine).astype(int)
            else: # If median is 0, any positive sunshine is "long" relative to median
                df_weather['is_long_sunshine'] = (df_weather['sunshine_duration'] > 0).astype(int)
            print(f"Categorized 'sunshine_duration' based on median: {median_sunshine:.2f} (1 for long, 0 for short)")
        else:
            df_weather['is_long_sunshine'] = 0
            print("Warning: 'sunshine_duration' not found in weather data for categorization.")
    else:
        print("Warning: No weather data for the period 2013 to calculate medians for duration features. Defaulting binary flags to 0.")
        df_weather['is_long_daylight'] = 0
        df_weather['is_long_sunshine'] = 0
    # --- END NEW ---

    if not df.empty and 'pickup_datetime' in df.columns and df_weather is not None:
        df['pickup_date_naive_for_merge'] = df['pickup_datetime'].dt.normalize().dt.tz_localize(None) 
        df_weather['weather_date_for_merge'] = pd.to_datetime(df_weather['date']).dt.normalize() 
        
        weather_cols_to_merge = [col for col in df_weather.columns if col not in ['date', 'weather_date_for_merge']]
        weather_cols_to_merge.append('weather_date_for_merge') 
        
        df = pd.merge(df, df_weather[weather_cols_to_merge], 
                      left_on='pickup_date_naive_for_merge', 
                      right_on='weather_date_for_merge', 
                      how='left')
        
        df.drop(columns=['pickup_date_naive_for_merge', 'weather_date_for_merge'], inplace=True, errors='ignore')
        df.drop(columns=['date_y'], inplace=True, errors='ignore') 
        if 'date_x' in df.columns and 'pickup_date' not in df.columns: 
            df.rename(columns={'date_x': 'pickup_date'}, inplace=True)
        print("Weather data merged with taxi data.")
        
        weather_features_actually_in_df = [
            'had_rain', 'had_snow_fall', 'is_extreme_temp', 
            'temperature_2m_mean', 'daylight_duration', 'sunshine_duration',
            'is_long_daylight', 'is_long_sunshine' # Added new binary features
        ]
        weather_features_actually_in_df = [col for col in weather_features_actually_in_df if col in df.columns]

        if weather_features_actually_in_df:
            print("NaNs in added weather features after merge:")
            print(df[weather_features_actually_in_df].isnull().sum())
            for w_col in weather_features_actually_in_df:
                if pd.api.types.is_numeric_dtype(df[w_col]):
                    df[w_col] = df[w_col].fillna(0)
                else:
                    df[w_col] = df[w_col].fillna("Unknown")
    else:
        print("Skipping weather data merge: main df is empty, 'pickup_datetime' is missing, or weather data not loaded.")

except FileNotFoundError: print(f"Error: Weather data file not found at {WEATHER_DATA_PATH}. Skipping weather merge.")
except Exception as e_weather: print(f"An error occurred processing weather data: {e_weather}")
print("*" * 40)

# --- 0.75 Load taxi zones shapefile (download from NYC TLC; see Simulator/README.md) ---
gdf_zones = None
if gpd is not None:
    _zone_paths = [
        BASE_DIR / "taxi_zones" / "taxi_zones.shp",
        BASE_DIR / "taxi_zones.shp",
        BASE_DIR / "taxi_zones.zip",
    ]
    for zp in _zone_paths:
        if not zp.exists():
            continue
        try:
            cand = gpd.read_file(zp)
            required_zone_cols = ["LocationID", "borough", "zone"]
            if not all(col in cand.columns for col in required_zone_cols):
                print(
                    f"Warning: zones file {zp} missing columns {required_zone_cols}; "
                    f"found {cand.columns.tolist()}"
                )
                continue
            cand = cand.copy()
            cand["LocationID"] = cand["LocationID"].astype(int)
            gdf_zones = cand
            print(f"Loaded taxi zones from: {zp} (shape {gdf_zones.shape})")
            break
        except Exception as e_z:
            print(f"Could not load taxi zones from {zp}: {e_z}")
    if gdf_zones is None and not any(p.exists() for p in _zone_paths):
        print(
            "No taxi zone file found next to this script "
            "(expected taxi_zones/taxi_zones.shp, taxi_zones.shp, or taxi_zones.zip)."
        )
    elif gdf_zones is None:
        print("Taxi zone paths exist but none could be loaded as a valid zones layer.")
else:
    print("GeoPandas is not installed; skipping taxi zone shapefile load and spatial join.")

# --- 0.8 Convert Lat/Lon to LocationID, Borough, Zone using GeoPandas ---
print("\n--- 0.8 Converting Coordinates to LocationIDs, Boroughs, Zones via Spatial Join ---")
if gpd is not None and gdf_zones is not None and not df.empty and all(
    col in df.columns
    for col in ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
):
    try:
        # Create a temporary join key if df.index is not unique or suitable
        df_reset_for_join = df.reset_index()
        join_key_col = 'original_index_for_sjoin' # Name for the temporary index column
        df_reset_for_join.rename(columns={'index': join_key_col}, inplace=True)
        if join_key_col not in df_reset_for_join.columns: # If index was unnamed
            df_reset_for_join[join_key_col] = df_reset_for_join.index


        gdf_trips_pickup = gpd.GeoDataFrame(
            df_reset_for_join,
            geometry=gpd.points_from_xy(
                df_reset_for_join["pickup_longitude"], df_reset_for_join["pickup_latitude"]
            ),
            crs="EPSG:4326",
        )
        gdf_trips_dropoff = gpd.GeoDataFrame(
            df_reset_for_join,
            geometry=gpd.points_from_xy(
                df_reset_for_join["dropoff_longitude"], df_reset_for_join["dropoff_latitude"]
            ),
            crs="EPSG:4326",
        )
        # print("Created GeoDataFrames for trip pickup and dropoff points.")

        if gdf_trips_pickup.crs != gdf_zones.crs:
            # print(f"Aligning CRS: Zones ({gdf_zones.crs}) to Trips ({gdf_trips_pickup.crs})")
            gdf_zones_aligned = gdf_zones.to_crs(gdf_trips_pickup.crs) 
        else:
            gdf_zones_aligned = gdf_zones.copy()

        gdf_zones_for_join = gdf_zones_aligned[['LocationID', 'borough', 'zone', 'geometry']].copy()
        
        # print("Performing spatial join for pickup locations...")
        gdf_pickup_joined = gpd.sjoin(gdf_trips_pickup[[join_key_col, 'geometry']], 
                                      gdf_zones_for_join, 
                                      how='left', predicate='within')
        
        # Merge pickup zone info
        df = pd.merge(df_reset_for_join, 
                      gdf_pickup_joined[[join_key_col, 'LocationID', 'borough', 'zone']],
                      on=join_key_col, 
                      how='left')
        df.rename(columns={'LocationID': 'PULocationID',
                           'borough': 'pickup_borough',
                           'zone': 'pickup_zone'}, inplace=True)
        # print("Added PULocationID, pickup_borough, pickup_zone.")

        # print("Performing spatial join for dropoff locations...")
        gdf_dropoff_joined = gpd.sjoin(gdf_trips_dropoff[[join_key_col, 'geometry']], 
                                       gdf_zones_for_join, 
                                       how='left', predicate='within')
        
        # Merge dropoff zone info, careful with suffixes if columns already exist from pickup merge
        dropoff_attrs = gdf_dropoff_joined[[join_key_col, 'LocationID', 'borough', 'zone']].copy()
        dropoff_attrs.rename(columns={'LocationID': 'DOLocationID',
                                      'borough': 'dropoff_borough',
                                      'zone': 'dropoff_zone'}, inplace=True)
        df = pd.merge(df, dropoff_attrs, on=join_key_col, how='left')
        
        if join_key_col in df.columns: # Drop the temporary join key
            df.drop(columns=[join_key_col], inplace=True)
        
        # Restore original index if it was meaningful and not just a range index
        # If df_orig_index was a simple RangeIndex(start=0, stop=len(df_orig_index), step=1),
        # the current RangeIndex after merges (if no duplicates in join_key_col) is fine.
        # If it had a specific name or values, you might want to restore it.
        # For now, assume the new RangeIndex is acceptable.

        for col in ['PULocationID', 'DOLocationID']:
            if col in df.columns: df[col] = df[col].fillna(0).astype(int)
        for col in ['pickup_borough', 'pickup_zone', 'dropoff_borough', 'dropoff_zone']:
            if col in df.columns: df[col] = df[col].fillna('Unknown') # Corrected from inplace=True
        print("Added/Updated PULocationID, pickup_borough/zone, DOLocationID, dropoff_borough/zone.")
        if not df.empty:
             preview_cols = [c for c in ['PULocationID', 'pickup_borough', 'DOLocationID', 'dropoff_borough'] if c in df.columns]
             if preview_cols: print("Sample with new LocationIDs and Boroughs:\n", df[preview_cols].head())

    except Exception as e_geospatial: print(f"Error during geospatial processing: {e_geospatial}")
elif gpd is None:
    print("GeoPandas not installed; skipping spatial join.")
elif gdf_zones is None:
    print("Taxi zones shapefile not loaded; skipping spatial join.")
else:
    print("DataFrame empty or missing pickup/dropoff lat-lon columns. Skipping LocationID derivation.")
print("*" * 40)


# --- 1. Day-Level Anomalies ---
# ... (as before)
print("\n--- 1. Identify and Handle Day-Level Anomalies ---")
if not df.empty and 'pickup_datetime' in df.columns: 
    df['pickup_date'] = df['pickup_datetime'].dt.date
    daily_trips = df.groupby('pickup_date').size().reset_index(name='trip_count')
    if not daily_trips.empty:
        mean_trips = daily_trips['trip_count'].mean(); std_trips = daily_trips['trip_count'].std(); threshold = 5
        lower_bound = mean_trips - threshold * std_trips; upper_bound = mean_trips + threshold * std_trips
        anomalous_days_df = daily_trips[(daily_trips['trip_count'] > upper_bound) | (daily_trips['trip_count'] < lower_bound)].copy()
        df['is_anomaly_day'] = 0; anomalous_dates_set = set(anomalous_days_df['pickup_date'])
        df.loc[df['pickup_date'].isin(anomalous_dates_set), 'is_anomaly_day'] = 1
        print(f"Found {len(anomalous_days_df)} anomalous days. 'is_anomaly_day' column created.")
    else: print("No daily trips to analyze. Skipping day-level anomaly detection.")
else: print("DataFrame empty or 'pickup_datetime' missing. Skipping day-level anomaly detection.")
print("*" * 40)

# --- 2. Temporal Features ---
# ... (as before)
print("\n--- 2. Extracting Temporal Features ---")
if not df.empty and 'pickup_datetime' in df.columns:
    df['pickup_day_of_year'] = df['pickup_datetime'].dt.dayofyear
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.weekday  
    df['pickup_hour_of_day'] = df['pickup_datetime'].dt.hour 
    df['pickup_minute_of_hour'] = df['pickup_datetime'].dt.minute 
    df['pickup_second_of_minute'] = df['pickup_datetime'].dt.second 
    print("Temporal features extracted.")
    if not df.empty: print(df[['pickup_day_of_year', 'pickup_day_of_week', 'pickup_hour_of_day', 'pickup_minute_of_hour', 'pickup_second_of_minute']].head())
else: print("DataFrame empty or 'pickup_datetime' missing. Skipping temporal features.")
print("*" * 40)

# --- 3. Haversine Distance ---
# ... (as before, result in 'distance_mi') ...
print("\n--- 3. Calculating Haversine Distance (in Miles) ---")
def haversine_np_vectorized_miles(lat1_series, lon1_series, lat2_series, lon2_series):
    R = 3958.8; lat1, lon1, lat2, lon2 = map(np.radians, [lat1_series, lon1_series, lat2_series, lon2_series])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)); distance = R * c
    return distance
if not df.empty and all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']):
    df['distance_mi'] = haversine_np_vectorized_miles(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    print("Haversine distance ('distance_mi') calculated in miles.")
    df = df[df['distance_mi'] > 0.01].copy()
    print("Applied distance cleaning (removed zero/near-zero mile distances).")
    if not df.empty: print(df[['distance_mi']].head())
    else: print("DataFrame became empty after distance cleaning.")
elif df.empty: print("DataFrame is empty. Skipping Haversine distance calculation.")
else: print("Error: Lat/lon columns missing for distance_mi.")
print("*" * 40)

# --- 3.5 Business District & Airport Proximity (after LocationIDs/Zones derived) ---
print("\n--- 3.5 Creating Business District & Airport Proximity Features ---")
if not df.empty:
    # Airport Proximity
    airports_coords = {'JFK': (40.6413, -73.7781), 'LGA': (40.7769, -73.8740), 'EWR': (40.6895, -74.1745)}
    radius_mi = 0.5 
    def haversine_to_point_vectorized(lat_series, lon_series, point_lat, point_lon, unit='mi'):
        """
        Calculates distance from a series of points to a single point.
        Defaults to miles.
        """
        # Return an empty series with the correct index if the input is empty
        if lat_series.empty or lon_series.empty:
            return pd.Series(dtype='float64', index=lat_series.index)
        # Prepare data for vector calculation, dropping any rows with missing coordinates
        valid_indices = lat_series.notna() & lon_series.notna()
        if not valid_indices.any():
            return pd.Series(np.nan, dtype='float64', index=lat_series.index)
        series_coords_array = np.array(list(zip(lat_series[valid_indices], lon_series[valid_indices])))
        points_array = np.array([(point_lat, point_lon)] * len(series_coords_array))
        if len(series_coords_array) == 0:
            return pd.Series(np.nan, dtype='float64', index=lat_series.index)
        try:
            from haversine import haversine_vector, Unit
            dist_unit_map = {
                'mi': Unit.MILES,
                'km': Unit.KILOMETERS,
                'm': Unit.METERS
            }
            # CORRECTED LOGIC: Get the unit from the map, and default to MILES if the unit is not found.
            selected_unit = dist_unit_map.get(unit.lower(), Unit.MILES)
            if unit.lower() not in dist_unit_map:
                print(f"Warning: Invalid unit '{unit}'. Defaulting to miles.")
            # Calculate distances for valid coordinates
            distances_valid = haversine_vector(series_coords_array, points_array, unit=selected_unit)
            # Create a full series with NaNs and place the valid distances back
            distances_full = pd.Series(np.nan, index=lat_series.index)
            distances_full.loc[valid_indices] = distances_valid
            return distances_full  
        except ImportError:
            print("Warning: 'haversine' library not found. Cannot calculate distances.")
    
    df['pickup_near_airport'] = 0; df['dropoff_near_airport'] = 0
    for airport_name, coords in airports_coords.items():
        # print(f"Calculating proximity to {airport_name} (radius: {radius_mi} mile)...") # Verbose
        if all(c in df.columns for c in ['pickup_latitude', 'pickup_longitude']):
            dist_pickup = haversine_to_point_vectorized(df['pickup_latitude'], df['pickup_longitude'], coords[0], coords[1], unit='mi')
            if dist_pickup is not None: df.loc[dist_pickup <= radius_mi, 'pickup_near_airport'] = 1
        if all(c in df.columns for c in ['dropoff_latitude', 'dropoff_longitude']):
            dist_dropoff = haversine_to_point_vectorized(df['dropoff_latitude'], df['dropoff_longitude'], coords[0], coords[1], unit='mi')
            if dist_dropoff is not None: df.loc[dist_dropoff <= radius_mi, 'dropoff_near_airport'] = 1
    print("Airport proximity features created.")
    if not df.empty: print("Pickup near airport counts:\n", df['pickup_near_airport'].value_counts()); print("Dropoff near airport counts:\n", df['dropoff_near_airport'].value_counts())

    # Business District (dependent on pickup_zone/PULocationID from step 0.8)
    business_zone_names_example = ['Midtown Center', 'Financial District North', 'Times Sq/Theatre District', 'Upper East Side South', 'Midtown East'] 
    business_district_ids = set() # User can populate this if preferring ID based e.g. {161, 162, ...}

    df['pickup_is_business_district'] = 0; df['dropoff_is_business_district'] = 0
    if 'pickup_zone' in df.columns: 
        df['pickup_is_business_district'] = df['pickup_zone'].apply(lambda x: 1 if x in business_zone_names_example else 0)
    elif 'PULocationID' in df.columns: 
         if not business_district_ids: print("Info: 'business_district_ids' is empty for pickup_is_business_district (ID-based).")
         df['pickup_is_business_district'] = df['PULocationID'].apply(lambda x: 1 if x in business_district_ids else 0)
    else: print("Warning: Could not create 'pickup_is_business_district' (missing pickup_zone/PULocationID).")
    if 'dropoff_zone' in df.columns:
        df['dropoff_is_business_district'] = df['dropoff_zone'].apply(lambda x: 1 if x in business_zone_names_example else 0)
    elif 'DOLocationID' in df.columns:
         if not business_district_ids: print("Info: 'business_district_ids' is empty for dropoff_is_business_district (ID-based).")
         df['dropoff_is_business_district'] = df['DOLocationID'].apply(lambda x: 1 if x in business_district_ids else 0)
    else: print("Warning: Could not create 'dropoff_is_business_district' (missing dropoff_zone/DOLocationID).")
    print("Created 'pickup_is_business_district' and 'dropoff_is_business_district'.")
    if not df.empty and 'pickup_is_business_district' in df.columns: print("Pickup in business district counts:\n", df['pickup_is_business_district'].value_counts())
else: print("DataFrame is empty. Skipping business district & airport proximity features.")
print("*" * 40)

# --- 3.6 Trip Directionality Features (NEW SECTION) ---
print("\n--- 3.6 Creating Trip Directionality Features ---")
if not df.empty and 'pickup_hour_of_day' in df.columns and \
   all(f in df.columns for f in ['pickup_is_business_district', 'dropoff_is_business_district', 'pickup_near_airport', 'dropoff_near_airport']):

    # Define residential proxy (not business and not airport)
    # Residential proxy: no direct residential label exists, so approximate as
    # neither business district nor airport (non-core / residential-like area).
    df['is_pickup_residential_proxy'] = ((df['pickup_is_business_district'] == 0) & (df['pickup_near_airport'] == 0)).astype(int)
    df['is_dropoff_residential_proxy'] = ((df['dropoff_is_business_district'] == 0) & (df['dropoff_near_airport'] == 0)).astype(int)

    # Morning commute: Residential -> Business District (e.g., 6-9 AM UTC) 住宅区到商业区的早高峰通勤
    is_morning_rush = (df['pickup_hour_of_day'] >= 6) & (df['pickup_hour_of_day'] <= 9)
    df['is_morning_commute'] = (is_morning_rush & (df['is_pickup_residential_proxy'] == 1) & (df['dropoff_is_business_district'] == 1)).astype(int)
    print("Created 'is_morning_commute' feature.")
    if not df.empty: print(df['is_morning_commute'].value_counts())

    # Evening commute: business district -> residential proxy (e.g., 5-7 PM UTC).
    is_evening_rush = (df['pickup_hour_of_day'] >= 17) & (df['pickup_hour_of_day'] <= 19)
    df['is_evening_commute'] = (is_evening_rush & (df['pickup_is_business_district'] == 1) & (df['is_dropoff_residential_proxy'] == 1)).astype(int)
    print("Created 'is_evening_commute' feature.")
    if not df.empty: print(df['is_evening_commute'].value_counts())

    # Business district to airport.
    df['is_biz_to_airport'] = ((df['pickup_is_business_district'] == 1) & (df['dropoff_near_airport'] == 1)).astype(int)
    print("Created 'is_biz_to_airport' feature.")
    if not df.empty: print(df['is_biz_to_airport'].value_counts())
    
    # Airport to business district.
    df['is_airport_to_biz'] = ((df['pickup_near_airport'] == 1) & (df['dropoff_is_business_district'] == 1)).astype(int)
    print("Created 'is_airport_to_biz' feature.")
    if not df.empty: print(df['is_airport_to_biz'].value_counts())
else:
    print("DataFrame empty or prerequisite columns missing. Skipping trip directionality features.")
print("*" * 40)


# --- Trip-Level Anomaly Detection ---
# ... (as before) ...
print("\n--- Applying Trip-Level Anomaly Detection (Isolation Forest on distance_mi & fare_amount) ---")
if not df.empty and 'fare_amount' in df.columns and 'distance_mi' in df.columns:
    df_for_iso = df[['distance_mi', 'fare_amount']].dropna()
    if not df_for_iso.empty and len(df_for_iso) > 1 :
        clf = IsolationForest(random_state=42, contamination=0.01)
        predictions = clf.fit_predict(df_for_iso)
        df.loc[df_for_iso.index, 'Anomaly_trip_fare'] = predictions
        shape_before_iso = df.shape; df = df.loc[df['Anomaly_trip_fare'] == 1].copy()
        df.drop('Anomaly_trip_fare', axis=1, inplace=True, errors='ignore')
        print(f"Shape before Isolation Forest: {shape_before_iso}, after: {df.shape}"); print("Trip-level anomalies removed.")
    elif len(df_for_iso) <=1: print("Not enough data for Isolation Forest.")
    else: print("No data for Isolation Forest (df_for_iso is empty).")
elif df.empty: print("DataFrame is empty. Skipping trip-level anomaly detection.")
else: print("Warning: 'fare_amount' or 'distance_mi' not available for trip-level anomaly detection.")
print("*" * 40)

# --- 4. Define Traffic Congestion (Rule-Based) ---
# ... (as before) ...
print("\n--- 4. Defining Traffic Congestion (Rule-Based) ---")
if not df.empty and 'pickup_hour_of_day' in df.columns and 'pickup_day_of_week' in df.columns:
    df['traffic_congestion'] = 0
    congestion_condition = ((df['pickup_hour_of_day'] >= 8) & (df['pickup_hour_of_day'] <= 18) & (df['pickup_day_of_week'] >= 0) & (df['pickup_day_of_week'] <= 4))
    df.loc[congestion_condition, 'traffic_congestion'] = 1
    print("Rule-based traffic congestion feature created.")
else: print("DataFrame empty or time columns missing. Skipping traffic congestion.")
print("*" * 40)

# --- Passenger Count Analysis (Final check after all filtering) ---
# ... (as before) ...
print("\n--- Passenger Count Analysis (Final Check) ---")
if 'passenger_count' in df.columns and not df.empty:
    print("Passenger counts distribution (after all filters including 1-6 range):")
    print(df['passenger_count'].value_counts().sort_index())
else: print("Passenger count column not available or df empty.")
print("*" * 40)

# --- Section 5: Airport Proximity (already created in 3.5) ---
print("\n--- 5. Airport Proximity Features (Created in Section 3.5) ---")
if not df.empty and 'pickup_near_airport' in df.columns:
    print("Pickup near airport counts:\n", df['pickup_near_airport'].value_counts())
    print("Dropoff near airport counts:\n", df['dropoff_near_airport'].value_counts())
else:
    print("Airport proximity features not available or df empty.")
print("*" * 40)


# --- Section 6: Manhattan Boundaries Feature (Created by GeoPandas if PULocationID was derived) ---
print("\n--- 6. NYC (Manhattan) Boundaries Feature ---")
# This was intended to be based on lat/lon directly, not PULocationID.
# Re-creating it here based on lat/lon to ensure it's always attempted if coords exist.
manhattan_bounds = {'lat_min': 40.70, 'lat_max': 40.88, 'lon_min': -74.02, 'lon_max': -73.90}
def is_within_manhattan_vectorized(lat_series, lon_series):
    return ((lat_series >= manhattan_bounds['lat_min']) & (lat_series <= manhattan_bounds['lat_max']) & \
            (lon_series >= manhattan_bounds['lon_min']) & (lon_series <= manhattan_bounds['lon_max']))
if not df.empty:
    if all(c in df.columns for c in ['pickup_latitude', 'pickup_longitude']):
        df['pickup_in_manhattan'] = is_within_manhattan_vectorized(df['pickup_latitude'], df['pickup_longitude']).astype(int)
        if not df.empty: print("Pickup in Manhattan counts:\n", df['pickup_in_manhattan'].value_counts())
    else: df['pickup_in_manhattan'] = 0 # Default if coords missing
    
    if all(c in df.columns for c in ['dropoff_latitude', 'dropoff_longitude']):
        df['dropoff_in_manhattan'] = is_within_manhattan_vectorized(df['dropoff_latitude'], df['dropoff_longitude']).astype(int)
        if not df.empty: print("Dropoff in Manhattan counts:\n", df['dropoff_in_manhattan'].value_counts())
    else: df['dropoff_in_manhattan'] = 0 # Default if coords missing
    print("Manhattan boundary features created/updated.")
else: print("DataFrame is empty. Skipping Manhattan boundary.")
print("*" * 40)



# --- 7. Analyze NYC Map and Improve Distance (OSMnx - on a sample/single row) ---
print("\n--- 7. OSMnx Road Distance Calculation (Example for one trip) ---")
if not df.empty:
    try:
        print("Loading street graph for Manhattan (this may take a while if first time)...")
        G = ox.graph_from_place('Manhattan, New York City, New York, USA', network_type='drive_service')
        print("Street graph loaded.")
        distance_col_for_osmnx = 'distance_mi' 
        if distance_col_for_osmnx in df.columns and df[distance_col_for_osmnx].notna().any():
            # Ensure pickup_in_manhattan and dropoff_in_manhattan exist before using in .get()
            df['pickup_in_manhattan'] = df.get('pickup_in_manhattan', 0) 
            df['dropoff_in_manhattan'] = df.get('dropoff_in_manhattan', 0)

            example_row_candidates = df[(df[distance_col_for_osmnx] > 0.1) & (df['pickup_in_manhattan'] == 1) & (df['dropoff_in_manhattan'] == 1)].head(1)
            if example_row_candidates.empty: example_row_candidates = df[df[distance_col_for_osmnx] > 0.1].head(1)
            if not example_row_candidates.empty:
                row = example_row_candidates.iloc[0]
                key_col_name = 'key' if 'key' in row.index else ('id' if 'id' in row.index else 'N/A_ID')
                print(f"\nCalculating road distance for example trip: {row.get(key_col_name, 'N/A')}")
                print(f"Haversine distance: {row[distance_col_for_osmnx]:.2f} mile")
                pickup_node = ox.distance.nearest_nodes(G, X=row['pickup_longitude'], Y=row['pickup_latitude'])
                dropoff_node = ox.distance.nearest_nodes(G, X=row['dropoff_longitude'], Y=row['dropoff_latitude'])
                if pickup_node != dropoff_node:
                    try: 
                        route_nodes = ox.shortest_path(G, pickup_node, dropoff_node, weight="length")
                        if route_nodes:
                            route_length_meters = 0
                            # Iterate over pairs of nodes in the route to get edge lengths
                            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                                # For MultiDiGraph, get_edge_data returns a dict of edge data dicts, keyed by edge index (usually 0 for shortest path)
                                edge_data = G.get_edge_data(u, v)
                                if edge_data: # Check if edge exists
                                    # If multiple edges, shortest_path usually picks one. We take the first one's length.
                                    route_length_meters += edge_data[0].get('length', 0)
                            route_length_km = route_length_meters / 1000.0
                            print(f"OSMnx road distance: {route_length_km:.2f} km")
                        else: print("No OSMnx route found.")
                    except Exception as e_route: print(f"Error calculating OSMnx route length: {e_route}")
                else: print("Pickup and dropoff nodes are the same or too close for OSMnx routing.")
            else: print("No suitable example row found for OSMnx.")
        else: print(f"'{distance_col_for_osmnx}' column missing/empty for OSMnx example.")
    except Exception as e: print(f"Error during OSMnx processing: {e}. Skipping.")
else: print("DataFrame is empty. Skipping OSMnx example.")
print("*" * 40)

# --- 8. Review Geospatial Location (Based on Manhattan filter - No Deletion) ---
# ... (as before) ...
print("\n--- 8. Reviewing Trips Based on Manhattan Boundaries (No Deletion) ---")
if not df.empty:
    initial_rows = df.shape[0]
    if 'pickup_in_manhattan' in df.columns and 'dropoff_in_manhattan' in df.columns:
        print("Trips have been flagged for pickup/dropoff in Manhattan in Section 6.")
        print("No rows will be removed based on these flags in this step as per new requirement.")
    else: print("Manhattan boundary features not created or df empty. No review/filtering based on them here.")
    print(f"Shape of df remains: {df.shape} (No rows removed in this step for Manhattan boundaries)")
else: print("DataFrame is empty. Skipping Manhattan boundary review.")
print("*" * 40)

# --- 9. Analyze and Clean Fare Amount ---
# ... (as before) ...
print("\n--- 9. Analyzing Fare Amount (Log Transform and Quantile Filter) ---")
if not df.empty and 'fare_amount' in df.columns:
    df['fare_amount_log'] = np.log1p(df['fare_amount'])
    if not df['fare_amount_log'].empty:
        log_fare_min_q = df['fare_amount_log'].quantile(0.005)
        log_fare_max_q = df['fare_amount_log'].quantile(0.995)
        if log_fare_min_q < log_fare_max_q:
            print(f"Further filtering fare_amount_log with quantiles: Min={log_fare_min_q:.2f}, Max={log_fare_max_q:.2f}.")
            initial_rows_fare_q_filter = df.shape[0]
            df = df[(df['fare_amount_log'] >= log_fare_min_q) & (df['fare_amount_log'] <= log_fare_max_q)].copy()
            removed_rows_fare_q_filter = initial_rows_fare_q_filter - df.shape[0]
            print(f"Removed {removed_rows_fare_q_filter} trips based on quantile filter on log_fare_amount.")
        else: print("Quantile range for fare_amount_log invalid. Skipping quantile filtering.")
    else: print("fare_amount_log column empty. Skipping quantile filtering.")
elif df.empty: print("DataFrame is empty. Skipping fare amount analysis.")
else: print("Warning: 'fare_amount' column not found for cleaning.")
print("*" * 40)

# --- 10. Final Data Export ---
print("\n--- 10. Exporting Selected Features to CSV ---")
columns_to_export = [
    'fare_amount', 'passenger_count', 'distance_mi',
    'pickup_in_manhattan', 'dropoff_in_manhattan',
    'pickup_near_airport', 'dropoff_near_airport',
    # 'pickup_day_of_week', 'pickup_hour_of_day',
    # 'pickup_minute_of_hour', 'pickup_second_of_minute', 
    'traffic_congestion', 
    # 'pickup_borough', 'pickup_zone', 
    # 'dropoff_borough', 'dropoff_zone', 
    'pickup_is_business_district', 'dropoff_is_business_district',
    'is_pickup_residential_proxy', 'is_dropoff_residential_proxy',
    'is_morning_commute', 'is_evening_commute', # Added new directionality features
    'is_biz_to_airport', 'is_airport_to_biz',   # Added new directionality features
    'had_rain',         
    'had_snow_fall',            
    'is_extreme_temp',
    'is_long_daylight', 'is_long_sunshine'
]
# if 'pickup_datetime' in df.columns and 'pickup_datetime' not in columns_to_export:
#     columns_to_export.append('pickup_datetime') 

existing_columns_to_export = [col for col in columns_to_export if col in df.columns]
if not df.empty and existing_columns_to_export:
    if len(existing_columns_to_export) < len(columns_to_export):
        missing_cols = set(columns_to_export) - set(existing_columns_to_export)
        print(f"Warning: CSV Exporter - Missing specified columns: {missing_cols}")
    df_exported = df[existing_columns_to_export].copy()
    output_path = BASE_DIR / "2013_4_Trained_nyc_taxi_data_with_directionality.csv"
    try:
        df_exported.to_csv(output_path, index=False)
        print(f"\nSaved selected features to: {output_path}")
        print(f"Exported CSV: {len(df_exported)} rows, {len(df_exported.columns)} columns.")
        print("Columns:", df_exported.columns.tolist())
    except Exception as e: print(f"\nError saving CSV: {e}")
elif df.empty: print("DataFrame is empty. Nothing to export to CSV.")
else: print("Error: none of the specified columns exist in df. Cannot export CSV.")

print("\n--- Full Data Processing Script (with Directionality Features & Sorting) Complete ---")
print("Final DataFrame shape:", df.shape)
if df.empty:
    print("DataFrame is empty after all processing.")
