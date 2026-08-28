import os
import numpy as np
import pandas as pd
from project_config import FUTURE_WINDOW

VALID_LAND_COVER = np.array([
    0, 1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240
])
SLOPE_BINS = [-45, -15, -10, -3, 3, 10, 15, 45]
SLOPE_LABELS = [-3, -2, -1, 0, 1, 2, 3]


def _timestamp_columns(df):
    filename = next((c for c in ["filename", "image_name", "name", "frame"] if c in df.columns), None)
    timestamp = next((c for c in ["timestamp_sec", "timestamp", "time"] if c in df.columns), None)
    if timestamp is None:
        raise ValueError(f"No timestamp column found. Columns: {df.columns.tolist()}")
    return filename, timestamp


def load_timestamp_map(path):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    filename_col, timestamp_col = _timestamp_columns(df)
    result = {}
    for idx, row in df.iterrows():
        name = os.path.basename(str(row[filename_col])) if filename_col else f"frame_{idx:06d}.png"
        result[name] = float(row[timestamp_col])
    return result


def robonav_score_series(robot_csv):
    df = pd.read_csv(robot_csv, sep=";")
    df.columns = df.columns.astype(str).str.strip()
    df["land_cover_class"] = df["land_cover"].apply(
        lambda x: VALID_LAND_COVER[np.abs(VALID_LAND_COVER - x).argmin()]
    )
    df["slope_deg"] = np.degrees(df["slope"])
    df["slope_class"] = pd.cut(
        df["slope_deg"], bins=SLOPE_BINS, labels=SLOPE_LABELS, include_lowest=True
    )
    terrain = (
        df.groupby(["land_cover_class", "slope_class"], observed=True)
        .agg(velocity=("navsat_vel", "mean"), smoothness=("smoothness_mean", "mean"))
        .reset_index()
    )
    velocity_max = terrain["velocity"].max()
    smoothness_max = terrain["smoothness"].max()
    terrain["velocity_cost"] = 1.0 - terrain["velocity"] / velocity_max
    terrain["smoothness_cost"] = terrain["smoothness"] / smoothness_max
    terrain["trav_cost"] = (terrain["velocity_cost"] + 0.5 * terrain["smoothness_cost"]).clip(0.0, 1.0)
    terrain["trav_score"] = 1.0 - terrain["trav_cost"]
    df = df.merge(
        terrain[["land_cover_class", "slope_class", "trav_score"]],
        on=["land_cover_class", "slope_class"], how="left"
    )
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df.dropna(subset=["time", "trav_score"]).sort_values("time")[["time", "trav_score"]]


def robonav_gt_map(robot_csv, timestamps_csv, future_window=FUTURE_WINDOW):
    scores = robonav_score_series(robot_csv)
    times = scores["time"].to_numpy()
    values = scores["trav_score"].to_numpy()
    timestamps = load_timestamp_map(timestamps_csv)
    result = {}
    for name, image_time in timestamps.items():
        idx = np.abs(times - (image_time + future_window)).argmin()
        score = float(values[idx])
        result[name] = score
        result[os.path.splitext(name)[0]] = score
    return result


def spectral_arc_length_band(signal, fs, f_low=20.0, f_high=35.0):
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 4:
        return np.nan
    signal = signal - np.mean(signal)
    nfft = int(2 ** np.ceil(np.log2(len(signal))) * 16)
    spectrum = np.abs(np.fft.rfft(signal, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= f_low) & (freqs <= f_high)
    f = freqs[mask]
    magnitude = spectrum[mask]
    if len(f) < 2 or np.max(magnitude) == 0:
        return np.nan
    magnitude = magnitude / np.max(magnitude)
    f_norm = (f - f[0]) / (f[-1] - f[0])
    return -np.sum(np.sqrt(np.diff(f_norm) ** 2 + np.diff(magnitude) ** 2))


def rellis_smoothness(df_imu):
    df = df_imu.copy().sort_values("timestamp_sec")
    times = df["timestamp_sec"].to_numpy(dtype=np.float64)
    dt = np.diff(times)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt)
    if fs / 2.0 < 35.0:
        raise ValueError(f"IMU sampling rate too low for 20-35 Hz SPARC: {fs:.2f} Hz")
    rows = []
    expected = fs
    for t in np.arange(np.ceil(times.min()), np.floor(times.max()), 1.0):
        window = df[(df["timestamp_sec"] >= t) & (df["timestamp_sec"] < t + 1.0)]
        if len(window) < 0.8 * expected:
            continue
        values = [
            spectral_arc_length_band(window[f"angular_velocity_{axis}"].values, fs)
            for axis in ("x", "y", "z")
        ]
        rows.append({"time": t + 0.5, "smoothness_mean": np.nanmean(values)})
    return pd.DataFrame(rows)


def rellis_gt_map(imu_csv, odom_csv, timestamps_csv, future_window=FUTURE_WINDOW):
    imu = pd.read_csv(imu_csv)
    odom = pd.read_csv(odom_csv)
    ts = pd.read_csv(timestamps_csv)
    for df in (imu, odom, ts):
        df.columns = df.columns.astype(str).str.strip()
        df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
        df.dropna(subset=["timestamp_sec"], inplace=True)
    smooth = rellis_smoothness(imu)
    if "velocity" not in odom.columns:
        odom["velocity"] = np.sqrt(
            odom["linear_velocity_x"] ** 2 + odom["linear_velocity_y"] ** 2 + odom["linear_velocity_z"] ** 2
        )
    odom["second"] = np.floor(odom["timestamp_sec"]).astype(np.int64)
    velocity = odom.groupby("second")["velocity"].mean().reset_index()
    velocity["time"] = velocity["second"] + 0.5
    merged = pd.merge_asof(
        smooth.sort_values("time"), velocity[["time", "velocity"]].sort_values("time"),
        on="time", direction="nearest", tolerance=0.6
    ).dropna(subset=["velocity", "smoothness_mean"])
    merged["roughness"] = -merged["smoothness_mean"]
    velocity_max = merged["velocity"].max()
    roughness_max = merged["roughness"].max()
    merged["velocity_cost"] = np.clip(1.0 - merged["velocity"] / velocity_max, 0.0, 1.0)
    merged["smoothness_cost"] = np.clip(merged["roughness"] / roughness_max, 0.0, 1.0)
    merged["trav_cost"] = np.clip(merged["velocity_cost"] + 0.5 * merged["smoothness_cost"], 0.0, 1.0)
    merged["trav_score"] = 1.0 - merged["trav_cost"]
    feature_times = merged["time"].to_numpy()
    feature_scores = merged["trav_score"].to_numpy()
    result = {}
    for _, row in ts.iterrows():
        name = os.path.basename(str(row["filename"]))
        idx = np.abs(feature_times - (float(row["timestamp_sec"]) + future_window)).argmin()
        score = float(feature_scores[idx])
        result[name] = score
        result[os.path.splitext(name)[0]] = score
    return result
