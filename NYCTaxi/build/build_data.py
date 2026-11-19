"""Build the cleaned request tables from the raw HVFHV files (2026-11-17).

One feature definition and one cleaning rule for every split:
  fit   : uniform sample of FIT_SAMPLE requests from FIT_MONTHS (simulator estimation)
  val   : all requests of the VAL_DAYS, in request order
  test  : all requests of the TEST_DAYS, in request order
Request times are recorded in whole seconds; requests recorded in the same second receive independent uniform
offsets within that second. Writes data/processed/{fit.parquet, cache/<split>_{t_ns,fare,X}.npy, summary.json}.

    python build_data.py
"""
import json
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
    out["is_extreme_temp"] = ((w.temperature_2m_max > 90) | (w.temperature_2m_min < 32)).astype(np.float32)  # deg F
    out["is_long_daylight"] = (w.daylight_duration >= 12 * 3600).astype(np.float32)
    return out.set_index("date")


def load_month(month, Z, W):
    """Cleaned requests of one calendar month: DataFrame with time, fare and FEATURES."""
    cols = ["hvfhs_license_num", "request_datetime", "pickup_datetime", "PULocationID", "DOLocationID",
            "trip_miles", "base_passenger_fare", "shared_request_flag"]
    d = pd.read_parquet(C.RAW / f"fhvhv_tripdata_{month}.parquet", columns=cols).rename(columns={"request_datetime": "time"})
    n_raw = len(d)
    lo = pd.Timestamp(month + "-01"); hi = lo + pd.offsets.MonthBegin(1)
    pu = d.PULocationID.to_numpy(); do = d.DOLocationID.to_numpy()
    ok = (d.hvfhs_license_num.isin(C.LICENSES) & (d.shared_request_flag == "N")
          & d.time.notna() & (d.time >= lo) & (d.time < hi) & (d.time <= d.pickup_datetime)
          & d.base_passenger_fare.between(C.PRICE_LO, C.PRICE_HI)
          & d.trip_miles.between(C.DIST_LO, C.DIST_HI)
          & (pu >= 1) & (pu <= 263) & (do >= 1) & (do <= 263)).to_numpy()
    ok &= Z["known"][np.clip(pu, 0, 265)] & Z["known"][np.clip(do, 0, 265)]
    d = d[ok]; pu = pu[ok]; do = do[ok]
    hour = d.time.dt.hour.to_numpy(); dow = d.time.dt.dayofweek.to_numpy()
    pb, db = Z["biz"][pu], Z["biz"][do]
    pa, da = Z["air"][pu], Z["air"][do]
    wk = W.reindex(d.time.dt.normalize()).to_numpy(np.float32)
    assert not np.isnan(wk).any()
    F = {
        "is_lyft": (d.hvfhs_license_num == "HV0005").to_numpy(),
        "distance_mi": d.trip_miles.to_numpy(),
        "pickup_in_manhattan": Z["man"][pu], "dropoff_in_manhattan": Z["man"][do],
        "pickup_near_airport": pa, "dropoff_near_airport": da,
        "traffic_congestion": (hour >= 8) & (hour <= 18) & (dow <= 4),
        "pickup_in_business_district": pb, "dropoff_in_business_district": db,
        "is_morning_commute": (hour >= 7) & (hour < 9) & ~pb & ~pa & db,
        "is_evening_commute": (hour >= 17) & (hour < 19) & pb & ~db & ~da,
        "is_biz_to_airport": pb & da, "is_airport_to_biz": pa & db,
        "had_rain": wk[:, 0], "had_snow_fall": wk[:, 1],
        "is_extreme_temp": wk[:, 2], "is_long_daylight": wk[:, 3],
        "is_weekend": dow >= 5, "is_night": (hour >= 22) | (hour < 5),
        "is_cross_borough": Z["boro"][pu] != Z["boro"][do],
    }
    out = pd.DataFrame({k: np.asarray(F[k], np.float32) for k in C.FEATURES})
    out.insert(0, "fare", d.base_passenger_fare.to_numpy(np.float32))
    out.insert(0, "time", d.time.to_numpy())
    return out, n_raw


def save_stream(df, name, rng):
    """Order a stream by request time after spreading requests within their second."""
    t = df.time.to_numpy("datetime64[ns]").astype(np.int64)
    t = t + rng.integers(0, 1_000_000_000, size=len(t))
    o = np.argsort(t, kind="stable")
    cache = C.PROC / "cache"; cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"{name}_t_ns.npy", t[o])
    np.save(cache / f"{name}_fare.npy", df.fare.to_numpy()[o])
    np.save(cache / f"{name}_X.npy", df[C.FEATURES].to_numpy(np.float32)[o])
    print(name, len(o), "requests", flush=True)


def days(d, list_of_days):
    keep = np.zeros(len(d), bool)
    for day in list_of_days:
        lo = pd.Timestamp(day); keep |= ((d.time >= lo) & (d.time < lo + pd.Timedelta(days=1))).to_numpy()
    return d[keep]


def main():
    C.PROC.mkdir(parents=True, exist_ok=True)
    Z, W = zone_table(), weather_table()
    rng = np.random.default_rng(C.SEED_DATA)
    fit, summary = [], {}
    for m in C.FIT_MONTHS:
        d, n_raw = load_month(m, Z, W); summary[m] = dict(raw=n_raw, clean=len(d))
        u = rng.random(len(d)).astype(np.float32)
        fit.append(d[u < 2.0 * C.FIT_SAMPLE / 200e6].assign(u=u[u < 2.0 * C.FIT_SAMPLE / 200e6]))
        print(m, n_raw, len(d), flush=True)
    fit = pd.concat(fit, ignore_index=True)
    fit = fit.nsmallest(C.FIT_SAMPLE, "u").drop(columns="u").sort_values("time").reset_index(drop=True)
    fit.to_parquet(C.PROC / "fit.parquet")
    print("fit sample", len(fit), "of", sum(v["clean"] for v in summary.values()), "cleaned requests", flush=True)
    for split, month, dd in (("val", C.VAL_MONTH, C.VAL_DAYS), ("test", C.TEST_MONTH, C.TEST_DAYS)):
        d, n_raw = load_month(month, Z, W); summary[month] = dict(raw=n_raw, clean=len(d))
        save_stream(days(d, dd), split, rng)
        print(month, n_raw, len(d), flush=True)
    (C.PROC / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
