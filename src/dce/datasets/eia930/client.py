"""
EIA-930 Power Grid Dataset Ingestion Client.

Handles downloading, parsing, and caching hourly power grid data for US Balancing Authorities.
Covers Demand (D), Net Generation (G), Wind (W), Solar (S), Net Interchange (I), and Forecasts.
Integrates with official Zenodo PUDL EIA-930 archives and metadata/ba_registry.csv.
"""

from typing import Dict, List, Optional, Tuple
import os
import json
import hashlib
import zipfile
import urllib.request
import numpy as np
import pandas as pd


# Major Balancing Authorities by Interconnection (Reference fallback dictionary)
BALANCING_AUTHORITIES = {
    # ERCOT (Texas)
    "ERCO": {"name": "Electric Reliability Council of Texas", "interconnection": "ERCOT", "rto": "ERCOT"},
    
    # CAISO & Western Interconnection (WECC)
    "CISO": {"name": "California Independent System Operator", "interconnection": "Western", "rto": "CAISO"},
    "BANC": {"name": "Balancing Authority of Northern California", "interconnection": "Western", "rto": "WECC_Other"},
    "BPAT": {"name": "Bonneville Power Administration", "interconnection": "Western", "rto": "WECC_Other"},
    "PACW": {"name": "PacifiCorp West", "interconnection": "Western", "rto": "WECC_Other"},
    "PACE": {"name": "PacifiCorp East", "interconnection": "Western", "rto": "WECC_Other"},
    "AZPS": {"name": "Arizona Public Service Company", "interconnection": "Western", "rto": "WECC_Other"},
    "NEVP": {"name": "Nevada Power Company", "interconnection": "Western", "rto": "WECC_Other"},
    "PSCO": {"name": "Public Service Company of Colorado", "interconnection": "Western", "rto": "WECC_Other"},
    "LDWP": {"name": "Los Angeles Department of Water and Power", "interconnection": "Western", "rto": "WECC_Other"},
    
    # Eastern Interconnection - RTOs & Vertically Integrated
    "PJM":  {"name": "PJM Interconnection", "interconnection": "Eastern", "rto": "PJM"},
    "MISO": {"name": "Midcontinent Independent System Operator", "interconnection": "Eastern", "rto": "MISO"},
    "SWPP": {"name": "Southwest Power Pool", "interconnection": "Eastern", "rto": "SPP"},
    "NYIS": {"name": "New York Independent System Operator", "interconnection": "Eastern", "rto": "NYISO"},
    "ISNE": {"name": "ISO New England", "interconnection": "Eastern", "rto": "ISONE"},
    "SOCO": {"name": "Southern Company Services", "interconnection": "Eastern", "rto": "SERC"},
    "TVA":  {"name": "Tennessee Valley Authority", "interconnection": "Eastern", "rto": "SERC"},
    "FPL":  {"name": "Florida Power & Light Company", "interconnection": "Eastern", "rto": "FRCC"},
    "DUK":  {"name": "Duke Energy Carolinas", "interconnection": "Eastern", "rto": "SERC"},
    "CPLE": {"name": "Duke Energy Progress East", "interconnection": "Eastern", "rto": "SERC"},
    "SC":   {"name": "South Carolina Public Service Authority", "interconnection": "Eastern", "rto": "SERC"},
}

ZENODO_EIA930_URLS = {
    "2021half1": "https://zenodo.org/api/records/22215263/files/eia930-2021half1.zip/content",
    "2021half2": "https://zenodo.org/api/records/22215263/files/eia930-2021half2.zip/content",
    "2022half2": "https://zenodo.org/api/records/22215263/files/eia930-2022half2.zip/content",
}


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_ba_registry(registry_path: Optional[str] = None) -> pd.DataFrame:
    """Load Balancing Authority registry metadata."""
    if registry_path is None:
        candidates = [
            os.path.join(os.getcwd(), "metadata", "ba_registry.csv"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "metadata", "ba_registry.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                registry_path = c
                break
    if registry_path and os.path.exists(registry_path):
        return pd.read_csv(registry_path)
    rows = []
    for ba, info in BALANCING_AUTHORITIES.items():
        rows.append({
            "ba_code": ba,
            "ba_name": info["name"],
            "interconnection": info["interconnection"],
            "rto_iso": info["rto"],
            "region": info["interconnection"],
            "timezone_offset_utc": -6,
            "status": "active"
        })
    return pd.DataFrame(rows)


def parse_raw_eia930_balance_zip(
    zip_path: str,
    ba_registry: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Extract and standardize hourly balance operations from a raw EIA-930 zip file.
    
    Extracts: demand, demand_forecast, generation, interchange, wind, solar, coal, gas, nuclear, hydro.
    """
    if ba_registry is None:
        ba_registry = load_ba_registry()
    valid_bas = set(ba_registry["ba_code"])
    ba_meta = ba_registry.set_index("ba_code").to_dict(orient="index")

    with zipfile.ZipFile(zip_path, "r") as z:
        balance_files = [n for n in z.namelist() if "balance" in n and n.endswith(".csv")]
        if not balance_files:
            raise ValueError(f"No balance CSV found in zip archive: {zip_path}")
        
        with z.open(balance_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)
            
    df = df[df["Balancing Authority"].isin(valid_bas)].copy()
    
    time_col = "UTC Time at End of Hour"
    if time_col in df.columns:
        df["timestamp"] = pd.to_datetime(df[time_col], utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["Data Date"] + " " + df["Hour Number"].astype(str) + ":00:00", utc=True)
        
    df["ba_code"] = df["Balancing Authority"]
    df["interconnection"] = df["ba_code"].map(lambda b: ba_meta[b]["interconnection"] if b in ba_meta else "Unknown")
    df["rto"] = df["ba_code"].map(lambda b: ba_meta[b]["rto_iso"] if b in ba_meta else "Unknown")

    def to_num(col_name: str) -> pd.Series:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name].astype(str).str.replace(",", ""), errors="coerce")
        return pd.Series(0.0, index=df.index)

    df["demand"] = to_num("Demand (MW) (Adjusted)").fillna(to_num("Demand (MW)"))
    df["demand_forecast"] = to_num("Demand Forecast (MW)")
    df["generation"] = to_num("Net Generation (MW) (Adjusted)").fillna(to_num("Net Generation (MW)"))
    df["interchange"] = to_num("Total Interchange (MW) (Adjusted)").fillna(to_num("Total Interchange (MW)"))
    df["wind"] = to_num("Net Generation (MW) from Wind").fillna(0.0)
    df["solar"] = to_num("Net Generation (MW) from Solar").fillna(0.0)
    df["coal"] = to_num("Net Generation (MW) from Coal").fillna(0.0)
    df["gas"] = to_num("Net Generation (MW) from Natural Gas").fillna(0.0)
    df["nuclear"] = to_num("Net Generation (MW) from Nuclear").fillna(0.0)
    df["hydro"] = to_num("Net Generation (MW) from Hydropower and Pumped Storage").fillna(0.0)

    cols = [
        "timestamp", "ba_code", "interconnection", "rto",
        "demand", "demand_forecast", "generation", "interchange",
        "wind", "solar", "coal", "gas", "nuclear", "hydro"
    ]
    return df[cols].sort_values(["timestamp", "ba_code"]).reset_index(drop=True)


def load_real_eia930_archive(
    period: str = "2021",
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    force_reprocess: bool = False
) -> pd.DataFrame:
    """
    Load real EIA-930 hourly dataset for specified period ('2021' or '2022h2').
    
    If processed parquet exists, loads directly. Otherwise extracts from data/raw/ zips.
    """
    parquet_path = os.path.join(processed_dir, f"eia930_{period}_hourly.parquet")
    if not force_reprocess and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    if period == "2021":
        zips = ["eia930-2021half1.zip", "eia930-2021half2.zip"]
    elif period == "2022h2":
        zips = ["eia930-2022half2.zip"]
    else:
        raise ValueError(f"Unsupported period: {period}. Use '2021' or '2022h2'.")
        
    ba_reg = load_ba_registry()
    dfs = []
    for zname in zips:
        zpath = os.path.join(raw_dir, zname)
        if not os.path.exists(zpath):
            key = zname.replace(".zip", "").replace("eia930-", "")
            if key in ZENODO_EIA930_URLS:
                print(f"Downloading {zname} from Zenodo archive...")
                urllib.request.urlretrieve(ZENODO_EIA930_URLS[key], zpath)
            else:
                raise FileNotFoundError(f"Missing raw EIA file: {zpath}")
        dfs.append(parse_raw_eia930_balance_zip(zpath, ba_registry=ba_reg))
        
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "ba_code"]).sort_values(["timestamp", "ba_code"]).reset_index(drop=True)
    combined.to_parquet(parquet_path, index=False)
    return combined


def generate_synthetic_eia930_benchmark(
    n_hours: int = 8760 * 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic EIA-930 benchmark dataset strictly for unit-test fixtures.
    
    [FIXTURE / SYNTHETIC TEST BENCHMARK ONLY]: Do not use for empirical manuscript claims.
    """
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range(start="2021-01-01 00:00:00", periods=n_hours, freq="h", tz="UTC")
    
    hours_of_day = timestamps.hour.values
    days_of_year = timestamps.dayofyear.values
    
    records = []
    solar_diurnal = np.maximum(0.0, np.sin((hours_of_day - 6) / 12.0 * np.pi)) ** 1.5
    wind_diurnal = 0.7 + 0.3 * np.cos((hours_of_day - 3) / 24.0 * 2 * np.pi)
    demand_base_curve = 0.8 + 0.3 * np.sin((hours_of_day - 7) / 24.0 * 2 * np.pi) + 0.15 * np.cos(days_of_year / 365.25 * 2 * np.pi)
    
    for ba_code, ba_info in BALANCING_AUTHORITIES.items():
        base_capacity = rng.uniform(8000, 45000)
        solar_capacity = base_capacity * rng.uniform(0.15, 0.40)
        wind_capacity = base_capacity * rng.uniform(0.15, 0.45)
        
        demand = base_capacity * demand_base_curve * rng.uniform(0.95, 1.05, size=n_hours)
        demand_forecast = demand * rng.uniform(0.96, 1.04, size=n_hours)
        solar = solar_capacity * solar_diurnal * rng.uniform(0.85, 1.0, size=n_hours)
        wind = wind_capacity * wind_diurnal * rng.weibull(2.0, size=n_hours) * 0.4
        
        if ba_code == "ERCO":
            uri_mask = (timestamps >= "2021-02-12") & (timestamps <= "2021-02-18")
            demand[uri_mask] *= 1.45
            wind[uri_mask] *= 0.20
            
        net_load = np.maximum(0.0, demand - (wind + solar))
        thermal_gen = net_load * rng.uniform(0.92, 1.08, size=n_hours)
        total_gen = thermal_gen + wind + solar
        interchange = demand - total_gen + rng.randn(n_hours) * (base_capacity * 0.01)
        
        df_ba = pd.DataFrame({
            "timestamp": timestamps,
            "ba_code": ba_code,
            "interconnection": ba_info["interconnection"],
            "rto": ba_info["rto"],
            "demand": demand,
            "demand_forecast": demand_forecast,
            "generation": total_gen,
            "wind": wind,
            "solar": solar,
            "interchange": interchange,
        })
        records.append(df_ba)
        
    df_all = pd.concat(records, ignore_index=True)
    return df_all

