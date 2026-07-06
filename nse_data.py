"""
nse_data.py — NSE Index Data Downloader
=========================================
Downloads NSE index data using BharatFinTrack as primary source.
Falls back to direct NSE API calls if BharatFinTrack fails
(e.g., when NSE changes their API endpoints).

Usage:
    from nse_data import download_tri, download_pri, download_index

    # Download TRI data (tries BharatFinTrack first, then direct API)
    df = download_tri("NIFTY 50", "01-Jan-2020", "01-Jul-2025")

    # Download PRI data (direct API only — BharatFinTrack has no PRI daily)
    df = download_pri("NIFTY 50", "01-Jan-2020", "01-Jul-2025")

    # Auto-detect and save to CSV
    df = download_index("NIFTY 50", data_type="TRI", csv_file="data/NIFTY 50.csv")
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# BharatFinTrack (primary)
# ──────────────────────────────────────────────
try:
    from BharatFinTrack import NSETRI, NSEProduct
    BFT_AVAILABLE = True
except ImportError:
    BFT_AVAILABLE = False
    log.warning("BharatFinTrack not installed. Using direct NSE API only.")


# ──────────────────────────────────────────────
# Direct NSE API (fallback)
# ──────────────────────────────────────────────
TRI_URL = "https://www.niftyindices.com/BackPage/getTotalReturnIndexString"
PRI_URL = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"

HTTP_HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://www.niftyindices.com/reports/historical-data",
    "Origin": "https://www.niftyindices.com",
    "X-Requested-With": "XMLHttpRequest",
}

MAX_DAYS_PER_REQUEST = 365


def _fetch_chunk(url: str, index_name: str, start_date: str, end_date: str, timeout: int = 20) -> list[dict]:
    """Make a single API call to NSE and return list of records."""
    params = {
        "name": index_name,
        "startDate": start_date,
        "endDate": end_date,
        "indexName": index_name,
    }
    payload = json.dumps({"cinfo": json.dumps(params)})

    resp = requests.post(url, data=payload, headers=HTTP_HEADERS, timeout=timeout)
    resp.raise_for_status()

    text = resp.text.strip()
    if not text or text[0] != "[":
        raise ValueError(f"NSE returned HTML instead of JSON. API may be down or index name invalid.")

    return resp.json()


def _date_range_chunks(start: datetime, end: datetime, chunk_days: int = MAX_DAYS_PER_REQUEST):
    """Split a date range into chunks of chunk_days."""
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _download_tri_direct(
    index_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Download TRI data directly from NSE API (fallback)."""
    fmt = "%d-%b-%Y"
    end_dt = datetime.now() if end_date is None else datetime.strptime(end_date, fmt)
    start_dt = (end_dt - timedelta(days=365 * 10)) if start_date is None else datetime.strptime(start_date, fmt)

    chunks = _date_range_chunks(start_dt, end_dt)
    all_records = []

    for i, (cs, ce) in enumerate(chunks):
        try:
            records = _fetch_chunk(TRI_URL, index_name, cs.strftime(fmt), ce.strftime(fmt))
            all_records.extend(records)
            if i < len(chunks) - 1:
                time.sleep(delay)
        except Exception as e:
            log.warning(f"  Chunk {cs.strftime(fmt)}-{ce.strftime(fmt)} failed: {e}")

    if not all_records:
        raise ValueError(f"No TRI data returned for {index_name}")

    df = pd.DataFrame(all_records)
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
    df["Close"] = pd.to_numeric(df["TotalReturnsIndex"].astype(str).str.replace(",", ""), errors="coerce")
    df = df[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    return df


def _download_tri_bft(
    index_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    csv_file: str | None = None,
) -> pd.DataFrame:
    """Download TRI data via BharatFinTrack (primary)."""
    tri = NSETRI()
    kwargs = {}
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date
    if csv_file:
        kwargs["csv_file"] = csv_file

    df = tri.download_daily_data(index=index_name, **kwargs)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────
def download_tri(
    index_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    csv_file: str | None = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Download Total Return Index (TRI) daily data.
    Tries BharatFinTrack first, falls back to direct NSE API.

    Parameters
    ----------
    index_name : str — e.g. "NIFTY 50", "NIFTY BANK"
    start_date : str — "dd-Mon-yyyy" e.g. "01-Jan-2015". Default: 10 years ago
    end_date : str — "dd-Mon-yyyy". Default: today
    csv_file : str — path to save CSV (optional)
    delay : float — seconds between API calls for fallback

    Returns DataFrame with columns: Date, Close
    """
    # Try BharatFinTrack first
    if BFT_AVAILABLE:
        try:
            df = _download_tri_bft(index_name, start_date, end_date, csv_file)
            if len(df) > 0:
                log.debug(f"  BharatFinTrack: {index_name} — {len(df)} rows")
                return df
        except Exception as e:
            log.debug(f"  BharatFinTrack failed for {index_name}: {e}, trying direct API...")

    # Fallback to direct NSE API
    df = _download_tri_direct(index_name, start_date, end_date, delay)

    if csv_file:
        Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_file, index=False)

    return df


def download_pri(
    index_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    csv_file: str | None = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Download Price Return Index (PRI) daily OHLC data from NSE.
    Direct API only — BharatFinTrack has no PRI daily download.

    Returns DataFrame with columns: Date, Open, High, Low, Close
    """
    fmt = "%d-%b-%Y"
    end_dt = datetime.now() if end_date is None else datetime.strptime(end_date, fmt)
    start_dt = (end_dt - timedelta(days=365 * 10)) if start_date is None else datetime.strptime(start_date, fmt)

    chunks = _date_range_chunks(start_dt, end_dt)
    all_records = []

    for i, (cs, ce) in enumerate(chunks):
        try:
            records = _fetch_chunk(PRI_URL, index_name, cs.strftime(fmt), ce.strftime(fmt))
            all_records.extend(records)
            if i < len(chunks) - 1:
                time.sleep(delay)
        except Exception as e:
            log.warning(f"  Chunk {cs.strftime(fmt)}-{ce.strftime(fmt)} failed: {e}")

    if not all_records:
        raise ValueError(f"No PRI data returned for {index_name}")

    df = pd.DataFrame(all_records)
    df["Date"] = pd.to_datetime(df["HistoricalDate"], format="%d %b %Y")

    for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    df = df.rename(columns={"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"})

    keep = ["Date"] + [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df = df[keep].dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if csv_file:
        Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_file, index=False)

    return df


def download_index(
    index_name: str,
    data_type: str = "TRI",
    start_date: str | None = None,
    end_date: str | None = None,
    csv_file: str | None = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Download TRI or PRI data. TRI tries BharatFinTrack first, then falls back to direct API.

    Parameters
    ----------
    index_name : str — e.g. "NIFTY 50"
    data_type : str — "TRI" or "PRI"
    start_date, end_date : str — "dd-Mon-yyyy" format
    csv_file : str — path to save CSV (optional)
    delay : float — seconds between API calls
    """
    if data_type.upper() == "TRI":
        return download_tri(index_name, start_date, end_date, csv_file, delay)
    else:
        return download_pri(index_name, start_date, end_date, csv_file, delay)


def download_all_indices(
    indices: list[str],
    data_type: str = "TRI",
    data_dir: str = "data",
    start_date: str | None = None,
    delay: float = 0.5,
    progress_callback=None,
) -> tuple[int, list[tuple[str, str]]]:
    """
    Download data for multiple indices. TRI tries BharatFinTrack first.

    Returns (success_count, [(failed_name, error_msg), ...])
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    failed = []
    succeeded = 0

    for i, idx in enumerate(indices):
        csv_path = str(Path(data_dir) / f"{idx}.csv")
        try:
            download_index(idx, data_type, start_date=start_date, csv_file=csv_path, delay=delay)
            succeeded += 1
        except Exception as e:
            failed.append((idx, str(e)))

        if progress_callback:
            progress_callback(i + 1, len(indices), idx)
        elif (i + 1) % 10 == 0:
            log.info(f"  Progress: {i+1}/{len(indices)}")

    return succeeded, failed


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 50)
    print("Testing NSE Data Downloader")
    print("=" * 50)

    if BFT_AVAILABLE:
        print(f"BharatFinTrack: installed (v{__import__('BharatFinTrack').__version__})")
    else:
        print("BharatFinTrack: NOT installed")

    print("\n1. Testing TRI (BharatFinTrack → fallback)...")
    try:
        df_tri = download_tri("NIFTY 50", "01-Jun-2025", "01-Jul-2025")
        print(f"   ✓ TRI: {len(df_tri)} rows")
        print(df_tri.head(3).to_string(index=False))
    except Exception as e:
        print(f"   ✗ TRI failed: {e}")

    print("\n2. Testing PRI (direct API)...")
    try:
        df_pri = download_pri("NIFTY 50", "01-Jun-2025", "01-Jul-2025")
        print(f"   ✓ PRI: {len(df_pri)} rows")
        print(df_pri.head(3).to_string(index=False))
    except Exception as e:
        print(f"   ✗ PRI failed: {e}")

    print("\n✓ Tests complete!")
