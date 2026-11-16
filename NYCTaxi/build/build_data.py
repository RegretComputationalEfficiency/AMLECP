"""Build the cleaned trip tables from the raw TLC files.

One feature definition and one cleaning rule for every split:
  fit   : uniform sample of FIT_SAMPLE trips from FIT_MONTHS (simulator estimation)
  val   : all trips in VAL_WINDOW, in arrival order
  test  : all trips in TEST_WINDOW, in arrival order
Pickup times are recorded in whole seconds. For the two streams, trips recorded in
the same second receive independent uniform offsets within that second.
"""
import numpy as np
import pandas as pd
import geopandas as gpd

import config as C


def zone_table():
    z = gpd.read_file(C.ZONES_DBF, ignore_geometry=True)[["LocationID", "borough", "zone"]]
    z = z.drop_duplicates("LocationID").set_index("LocationID")
    n = 266
    boro = np.full(n, "", dtype=object)
    man = np.zeros(n, bool); air = np.zeros(n, bool); biz = np.zeros(n, bool); known = np.zeros(n, bool)
    for i, r in z.iterrows():
        boro[i] = r.borough; known[i] = True
        man[i] = r.borough == "Manhattan"
        air[i] = i in C.AIRPORT_ZONE_IDS
        biz[i] = r.zone in C.BUSINESS_ZONES
    boro_id = pd.factorize(boro)[0]
    return dict(known=known, man=man, air=air, biz=biz, boro=boro_id)


def weather_table():
    w = pd.read_csv(C.WEATHER_CSV, parse_dates=["date"])
    out = pd.DataFrame({"date": w.date.dt.normalize()})
    out["had_rain"] = (w.rain_sum > 0).astype(np.float32)
    out["had_snow_fall"] = (w.snowfall_sum > 0).astype(np.float32)
    # temperatures in the file are in degrees Fahrenheit
    out["is_extreme_temp"] = ((w.temperature_2m_max > 90) | (w.temperature_2m_min < 32)).astype(np.float32)
    out["is_long_sunshine"] = (w.sunshine_duration >= 0.5 * w.daylight_duration).astype(np.float32)
    return out.set_index("date")


def load_month(month, Z, W):
    """Cleaned trips of one calendar month: DataFrame with time, fare and FEATURES."""
    parts = []
    for color, pre in (("yellow", "tpep"), ("green", "lpep")):
        f = C.RAW / f"{color}_tripdata_{month}.parquet"
        if not f.exists():
            continue
        cols = [f"{pre}_pickup_datetime", "passenger_count", "trip_distance", "RatecodeID",
                "PULocationID", "DOLocationID", "fare_amount"]
        d = pd.read_parquet(f, columns=cols).rename(columns={f"{pre}_pickup_datetime": "time"})
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    lo = pd.Timestamp(month + "-01"); hi = lo + pd.offsets.MonthBegin(1)
    pu = d.PULocationID.to_numpy(); do = d.DOLocationID.to_numpy()
    ok = ((d.time >= lo) & (d.time < hi) & (d.RatecodeID == 1)
          & d.fare_amount.between(C.PRICE_LO, C.PRICE_HI)
          & d.passenger_count.between(1, 6)
          & d.trip_distance.between(C.DIST_LO, C.DIST_HI)
          & (pu >= 1) & (pu <= 263) & (do >= 1) & (do <= 263)).to_numpy()
    ok &= Z["known"][np.clip(pu, 0, 265)] & Z["known"][np.clip(do, 0, 265)]
    d = d[ok]; pu = pu[ok]; do = do[ok]
    hour = d.time.dt.hour.to_numpy(); dow = d.time.dt.dayofweek.to_numpy()
    pb, db = Z["biz"][pu], Z["biz"][do]
    pa, da = Z["air"][pu], Z["air"][do]
    wk = W.reindex(d.time.dt.normalize()).to_numpy(np.float32)
    assert not np.isnan(wk).any()
    F = {
        "passenger_count": d.passenger_count.to_numpy(),
        "distance_mi": d.trip_distance.to_numpy(),
        "pickup_in_manhattan": Z["man"][pu], "dropoff_in_manhattan": Z["man"][do],
        "pickup_near_airport": pa, "dropoff_near_airport": da,
        "traffic_congestion": (hour >= 8) & (hour <= 18) & (dow <= 4),
        "pickup_in_business_district": pb, "dropoff_in_business_district": db,
        "is_morning_commute": (hour >= 7) & (hour < 9) & ~pb & ~pa & db,
        "is_evening_commute": (hour >= 17) & (hour < 19) & pb & ~db & ~da,
        "is_biz_to_airport": pb & da, "is_airport_to_biz": pa & db,
        "had_rain": wk[:, 0], "had_snow_fall": wk[:, 1],
        "is_extreme_temp": wk[:, 2], "is_long_sunshine": wk[:, 3],
        "is_weekend": dow >= 5, "is_night": (hour >= 22) | (hour < 5),
        "is_cross_borough": Z["boro"][pu] != Z["boro"][do],
    }
    out = pd.DataFrame({k: np.asarray(F[k], np.float32) for k in C.FEATURES})
    out.insert(0, "fare", d.fare_amount.to_numpy(np.float32))
    out.insert(0, "time", d.time.to_numpy())
    out["hour"] = hour.astype(np.int8); out["dow"] = dow.astype(np.int8)
    return out


def save_stream(df, name, rng):
    """Order a stream by arrival time after spreading trips within their second."""
    t = df.time.to_numpy("datetime64[ns]").astype(np.int64)
    t = t + rng.integers(0, 1_000_000_000, size=len(t))
    o = np.argsort(t, kind="stable")
    C.PROC.mkdir(parents=True, exist_ok=True)
    np.savez(C.PROC / f"{name}.npz", t_ns=t[o], fare=df.fare.to_numpy()[o],
             X=df[C.FEATURES].to_numpy(np.float32)[o])
    print(name, len(o), "arrivals")


def main():
    Z, W = zone_table(), weather_table()
    rng = np.random.default_rng(C.SEED_DATA)
    raw_rows = {}
    fit = []
    n_clean = 0
    months = {}
    for m in C.FIT_MONTHS:
        d = load_month(m, Z, W); months[m] = len(d); n_clean += len(d)
        fit.append(d.assign(u=rng.random(len(d)).astype(np.float32)))
        if m == "2014-11":
            v = d[(d.time >= C.VAL_WINDOW[0]) & (d.time < C.VAL_WINDOW[1])]
            save_stream(v, "val", rng)
        print(m, len(d), flush=True)
        # keep memory bounded: retain only a generous candidate set per month
        fit[-1] = fit[-1][fit[-1].u < 2.0 * C.FIT_SAMPLE / 160e6].copy()
    fit = pd.concat(fit, ignore_index=True)
    fit = fit.nsmallest(C.FIT_SAMPLE, "u").drop(columns="u").sort_values("time").reset_index(drop=True)
    fit.to_parquet(C.PROC / "fit.parquet")
    print("fit sample", len(fit), "of", n_clean, "cleaned trips")
    d = load_month("2014-12", Z, W)
    save_stream(d, "test", rng)
    pd.Series(months).to_csv(C.PROC / "clean_rows_by_month.csv")


if __name__ == "__main__":
    main()
