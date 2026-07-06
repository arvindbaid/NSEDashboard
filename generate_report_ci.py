"""
generate_report_ci.py — GitHub Actions compatible report generator
==================================================================
WORKFLOW:
    1. Data is downloaded LOCALLY by the user and committed to the repo
    2. GitHub Actions reads the committed CSV files, generates reports, and emails them

    This avoids NSE blocking cloud/datacenter IPs.

LOCAL DATA REFRESH (run before pushing):
    python generate_report_ci.py --download-only
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import logging
import os
import sys
import json
import urllib.request
import urllib.parse

from nse_data import download_all_indices as nse_download_all
import json as json_lib


def get_index_categories() -> dict:
    """Load index categories — try BharatFinTrack first, fallback to JSON file."""
    try:
        from BharatFinTrack import NSEProduct
        product = NSEProduct()
        return {cat: product.equity_categorical_indices(cat) for cat in CATEGORIES}
    except Exception:
        pass
    # Fallback: read from committed JSON
    json_path = Path(__file__).parent / "index_categories.json"
    if json_path.exists():
        with open(json_path) as f:
            return json_lib.load(f)
    log.error("No index categories available — neither BharatFinTrack nor index_categories.json found!")
    return {}

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECIPIENT_EMAILS = [e.strip() for e in os.environ.get("RECIPIENT_EMAILS", "").split(",") if e.strip()]
SKIP_EMAIL = os.environ.get("SKIP_EMAIL", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

CATEGORIES = ["broad", "sector", "thematic", "strategy"]
CATEGORY_LABELS = {"broad": "Broad Based", "sector": "Sectoral", "thematic": "Thematic", "strategy": "Strategy"}
BENCHMARK = "NIFTY 50"
ROLLING_PERIODS = [1, 3, 5, 10]
DATA_DIR = Path("data")
REPORT_DIR = Path("reports")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────
def download_all_data():
    """Download TRI data for all indices using new NSE API."""
    DATA_DIR.mkdir(exist_ok=True)
    idx_by_cat = get_index_categories()

    all_indices = set()
    for cat in CATEGORIES:
        all_indices.update(idx_by_cat.get(cat, []))
    all_indices.add(BENCHMARK)
    all_indices = sorted(all_indices)

    log.info(f"Downloading {len(all_indices)} indices from NSE...")
    succeeded, failed = nse_download_all(
        indices=all_indices,
        data_type="TRI",
        data_dir=str(DATA_DIR),
    )
    log.info(f"Download complete. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        for name, err in failed:
            log.warning(f"  ✗ {name}: {err}")
    return failed


def load_nav(index_name: str) -> pd.DataFrame | None:
    csv_path = DATA_DIR / f"{index_name}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
        return df.sort_values("Date").reset_index(drop=True)
    return None


# ──────────────────────────────────────────────
# CALCULATIONS
# ──────────────────────────────────────────────
def calc_yearly_returns(nav_df):
    nav_df = nav_df.set_index("Date").sort_index()
    yearly = {}
    for y in sorted(nav_df.index.year.unique()):
        y_start = nav_df[nav_df.index >= datetime(y, 1, 1)]
        if y + 1 <= datetime.now().year:
            y_end = nav_df[nav_df.index >= datetime(y + 1, 1, 1)]
            if len(y_start) > 0 and len(y_end) > 0:
                yearly[y] = round((y_end["Close"].iloc[0] / y_start["Close"].iloc[0] - 1) * 100, 2)
        else:
            if len(y_start) > 1:
                yearly[y] = round((y_start["Close"].iloc[-1] / y_start["Close"].iloc[0] - 1) * 100, 2)
    return yearly


def calc_monthly_returns(nav_df, months=12):
    nav_df = nav_df.set_index("Date").sort_index()
    results = {}
    today = nav_df.index.max()
    for i in range(months, 0, -1):
        m_end = today - pd.DateOffset(months=i - 1)
        m_start = today - pd.DateOffset(months=i)
        chunk = nav_df[(nav_df.index >= m_start) & (nav_df.index <= m_end)]
        if len(chunk) >= 2:
            ret = (chunk["Close"].iloc[-1] / chunk["Close"].iloc[0] - 1) * 100
            results[m_end.strftime("%b %Y")] = round(ret, 2)
    return results


def calc_period_returns(nav_df):
    nav_df = nav_df.set_index("Date").sort_index()
    latest = nav_df["Close"].iloc[-1]
    latest_date = nav_df.index[-1]
    results = {}
    for label, days in {"1M": 30, "3M": 90, "6M": 180, "9M": 270, "1Y": 365, "3Y": 1095, "5Y": 1825, "10Y": 3650}.items():
        target = latest_date - timedelta(days=days)
        past = nav_df[nav_df.index <= target]
        if len(past) > 0:
            past_val = past["Close"].iloc[-1]
            if days <= 365:
                results[label] = round((latest / past_val - 1) * 100, 2)
            else:
                results[label] = round(((latest / past_val) ** (1 / (days / 365.25)) - 1) * 100, 2)
    return results


def calc_rolling_returns(nav_df, period_years):
    nav_df = nav_df.set_index("Date").sort_index()
    close = nav_df["Close"]
    roll_days = int(period_years * 365.25)
    cagrs = []
    dates_list = close.index.tolist()
    for i in range(0, len(dates_list) - 1, 5):
        end_data = close[close.index >= dates_list[i] + timedelta(days=roll_days)]
        if len(end_data) == 0:
            break
        cagr = ((end_data.iloc[0] / close.iloc[i]) ** (1 / period_years) - 1) * 100
        cagrs.append(round(cagr, 2))
    if not cagrs:
        return {"avg": None, "min": None, "max": None, "prob_positive": None, "current": None}
    pos = sum(1 for c in cagrs if c > 0)
    return {
        "avg": round(np.mean(cagrs), 2), "min": round(min(cagrs), 2),
        "max": round(max(cagrs), 2), "prob_positive": round(pos / len(cagrs) * 100, 1),
        "current": cagrs[-1],
    }


# ──────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────
def generate_reports():
    REPORT_DIR.mkdir(exist_ok=True)
    idx_by_cat = get_index_categories()
    report_date = datetime.now().strftime("%Y-%m-%d")
    report_month = datetime.now().strftime("%B %Y")
    report_files = []

    # Check how many indices have data
    available = sum(1 for f in DATA_DIR.glob("*.csv") if f.stat().st_size > 100)
    if available == 0:
        log.error("No CSV data files found in data/ directory!")
        log.error("Run locally first: python generate_report_ci.py --download-only")
        log.error("Then: git add data/ && git commit -m 'Add index data' && git push")
        return []

    log.info(f"Found {available} index data files in data/")

    n50_nav = load_nav(BENCHMARK)
    n50_periods = calc_period_returns(n50_nav) if n50_nav is not None else {}

    for cat in CATEGORIES:
        indices = idx_by_cat.get(cat, [])
        cat_label = CATEGORY_LABELS[cat]
        filepath = REPORT_DIR / f"{cat}_report_{report_date}.xlsx"
        log.info(f"Generating: {cat_label} ({len(indices)} indices)...")

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            wb = writer.book
            hdr = wb.add_format({"bold": True, "bg_color": "#0f172a", "font_color": "#e2e8f0", "border": 1, "font_size": 11, "align": "center"})
            pos = wb.add_format({"font_color": "#16a34a", "num_format": "+0.00;-0.00", "bold": True})
            neg = wb.add_format({"font_color": "#dc2626", "num_format": "+0.00;-0.00", "bold": True})
            ttl = wb.add_format({"bold": True, "font_size": 14, "font_color": "#0ea5e9"})

            def write_sheet(df, sheet_name, title_text):
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
                ws = writer.sheets[sheet_name]
                ws.write(0, 0, title_text, ttl)
                for c, col_name in enumerate(df.columns):
                    ws.write(2, c, col_name, hdr)
                ws.set_column(0, 0, 38)
                ws.set_column(1, len(df.columns) - 1, 13)
                for c in range(1, len(df.columns)):
                    if df.dtypes.iloc[c] in [np.float64, np.int64, float, int]:
                        ws.conditional_format(3, c, len(df) + 2, c, {"type": "cell", "criteria": ">=", "value": 0, "format": pos})
                        ws.conditional_format(3, c, len(df) + 2, c, {"type": "cell", "criteria": "<", "value": 0, "format": neg})

            # Yearly
            rows = []
            for idx in indices:
                nav = load_nav(idx)
                if nav is not None:
                    yr = calc_yearly_returns(nav)
                    row = {"Index": idx}
                    row.update({str(k): v for k, v in yr.items()})
                    vals = list(yr.values())
                    row["Avg"] = round(np.mean(vals), 2) if vals else None
                    rows.append(row)
            if rows:
                df = pd.DataFrame(rows)
                year_cols = sorted([c for c in df.columns if c not in ["Index", "Avg"]], reverse=True)
                df = df[["Index"] + year_cols + ["Avg"]]
                write_sheet(df, "Yearly Returns", f"{cat_label} — Calendar Year Returns (%) | {report_month}")

            # Monthly
            rows = []
            m_labels = None
            for idx in indices:
                nav = load_nav(idx)
                if nav is not None:
                    mo = calc_monthly_returns(nav, 12)
                    if m_labels is None:
                        m_labels = list(mo.keys())
                    row = {"Index": idx}
                    row.update(mo)
                    row["12M Cum."] = round(sum(mo.values()), 2)
                    p = sum(1 for v in mo.values() if v > 0)
                    row["Hit Rate"] = f"{p}/{len(mo)}"
                    rows.append(row)
            if rows:
                df = pd.DataFrame(rows).sort_values("12M Cum.", ascending=False)
                write_sheet(df, "Monthly Returns", f"{cat_label} — Last 12 Months (%) | {report_month}")

            # Rolling
            for period in ROLLING_PERIODS:
                rows = []
                for idx in indices:
                    nav = load_nav(idx)
                    if nav is not None:
                        r = calc_rolling_returns(nav, period)
                        if r["avg"] is not None:
                            rows.append({
                                "Index": idx, "Current (%)": r["current"],
                                "Average (%)": r["avg"], "Min (%)": r["min"],
                                "Max (%)": r["max"], "Prob. Positive (%)": r["prob_positive"],
                            })
                if rows:
                    df = pd.DataFrame(rows).sort_values("Average (%)", ascending=False)
                    write_sheet(df, f"Rolling {period}Y", f"{cat_label} — {period}Y Rolling CAGR (%) | {report_month}")

            # vs NIFTY 50
            comp_periods = ["1M", "3M", "6M", "9M", "1Y"]
            rows = []
            for idx in indices:
                nav = load_nav(idx)
                if nav is not None:
                    pr = calc_period_returns(nav)
                    row = {"Index": idx}
                    oc = 0
                    for p in comp_periods:
                        own = pr.get(p)
                        n50v = n50_periods.get(p)
                        row[f"{p} Return"] = own
                        if own is not None and n50v is not None:
                            row[f"{p} Excess"] = round(own - n50v, 2)
                            if own > n50v:
                                oc += 1
                        else:
                            row[f"{p} Excess"] = None
                    row["Score"] = f"{oc}/{len(comp_periods)}"
                    rows.append(row)
            if rows:
                df = pd.DataFrame(rows).sort_values("1Y Excess", ascending=False, na_position="last")
                n50_str = " | ".join(f"{k}: {v:+.2f}%" for k, v in n50_periods.items() if k in comp_periods)
                write_sheet(df, "vs NIFTY 50", f"{cat_label} — vs NIFTY 50 [{n50_str}] | {report_month}")

        report_files.append(filepath)
        log.info(f"  ✓ {filepath.name}")

    # Summary
    summary_path = REPORT_DIR / f"summary_report_{report_date}.xlsx"
    log.info("Generating summary report...")
    all_rows = []
    for cat in CATEGORIES:
        indices = idx_by_cat.get(cat, [])
        for idx in indices:
            nav = load_nav(idx)
            if nav is not None:
                pr = calc_period_returns(nav)
                row = {"Index": idx, "Category": CATEGORY_LABELS[cat]}
                for p in ["1M", "3M", "6M", "1Y"]:
                    row[p] = pr.get(p)
                    n50v = n50_periods.get(p)
                    if pr.get(p) is not None and n50v is not None:
                        row[f"{p} Excess"] = round(pr[p] - n50v, 2)
                all_rows.append(row)

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        with pd.ExcelWriter(summary_path, engine="xlsxwriter") as writer:
            wb = writer.book
            hdr = wb.add_format({"bold": True, "bg_color": "#0f172a", "font_color": "#e2e8f0", "border": 1, "font_size": 11, "align": "center"})
            pos_f = wb.add_format({"font_color": "#16a34a", "num_format": "+0.00;-0.00", "bold": True})
            neg_f = wb.add_format({"font_color": "#dc2626", "num_format": "+0.00;-0.00", "bold": True})
            ttl = wb.add_format({"bold": True, "font_size": 14, "font_color": "#0ea5e9"})

            for sheet_name, df_sheet, _ in [
                ("Top 25 Outperformers", df_all.sort_values("1Y Excess", ascending=False).head(25), False),
                ("Bottom 25 Laggards", df_all.sort_values("1Y Excess", ascending=True).head(25), True),
            ]:
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
                ws = writer.sheets[sheet_name]
                ws.write(0, 0, f"{sheet_name} vs NIFTY 50 (1Y) | {report_month}", ttl)
                for c, cn in enumerate(df_sheet.columns):
                    ws.write(2, c, cn, hdr)
                ws.set_column(0, 0, 38)
                ws.set_column(1, 1, 14)
                ws.set_column(2, len(df_sheet.columns) - 1, 13)
                for c in range(2, len(df_sheet.columns)):
                    if df_sheet.dtypes.iloc[c] in [np.float64, np.int64, float, int]:
                        ws.conditional_format(3, c, len(df_sheet) + 2, c, {"type": "cell", "criteria": ">=", "value": 0, "format": pos_f})
                        ws.conditional_format(3, c, len(df_sheet) + 2, c, {"type": "cell", "criteria": "<", "value": 0, "format": neg_f})

        report_files.append(summary_path)
        log.info(f"  ✓ {summary_path.name}")
    else:
        log.warning("  No data for summary report — skipped")

    return report_files


# ──────────────────────────────────────────────
# EMAIL
# ──────────────────────────────────────────────
def send_email(report_files):
    report_date = datetime.now().strftime("%d %b %Y")
    report_month = datetime.now().strftime("%B %Y")

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECIPIENT_EMAILS)
    msg["Subject"] = f"NSE Index Monthly Report — {report_month}"

    body = f"""
    <html><body style="font-family: 'Segoe UI', Arial, sans-serif; color: #334155; line-height: 1.6;">
    <div style="max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #0ea5e9, #8b5cf6); padding: 20px 24px; border-radius: 12px 12px 0 0;">
            <h1 style="color: #fff; margin: 0; font-size: 22px;">₹ NSE Index Monthly Report</h1>
            <p style="color: rgba(255,255,255,0.8); margin: 4px 0 0;">{report_date}</p>
        </div>
        <div style="background: #f8fafc; padding: 20px 24px; border: 1px solid #e2e8f0; border-top: none; border-radius: 0 0 12px 12px;">
            <p>Your monthly NSE index analysis reports are attached.</p>
            <h3 style="color: #0ea5e9;">📎 Reports:</h3>
            <ul style="list-style: none; padding: 0;">
    """
    for f in report_files:
        body += f'<li style="padding: 4px 0;">📊 <strong>{f.name}</strong></li>\n'
    body += """
            </ul>
            <p style="font-size: 12px; color: #94a3b8; margin-top: 16px; border-top: 1px solid #e2e8f0; padding-top: 12px;">
                Data: NSE India TRI via BharatFinTrack | Generated via GitHub Actions</p>
        </div>
    </div></body></html>
    """
    msg.attach(MIMEText(body, "html"))

    for fp in report_files:
        if not fp.exists():
            continue
        with open(fp, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={fp.name}")
            msg.attach(part)

    # Auto-detect SMTP
    smtp_servers = {
        "gmail.com": ("smtp.gmail.com", 587),
        "hotmail.com": ("smtp-mail.outlook.com", 587),
        "outlook.com": ("smtp-mail.outlook.com", 587),
        "live.com": ("smtp-mail.outlook.com", 587),
        "yahoo.com": ("smtp.mail.yahoo.com", 587),
    }
    domain = SENDER_EMAIL.split("@")[-1].lower() if "@" in SENDER_EMAIL else ""
    smtp_host, smtp_port = smtp_servers.get(domain, ("smtp-mail.outlook.com", 587))

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls(context=context)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
    log.info(f"✓ Email sent to: {', '.join(RECIPIENT_EMAILS)}")


# ──────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────
def telegram_api(method, data=None, files=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    if files:
        boundary = "----TgBoundary"
        body = b""
        if data:
            for key, val in data.items():
                body += f"--{boundary}\r\n".encode()
                body += f"Content-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n".encode()
        for field, (fname, fbytes, ctype) in files.items():
            body += f"--{boundary}\r\n".encode()
            body += f"Content-Disposition: form-data; name=\"{field}\"; filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
            body += fbytes + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(url, data=body)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        body = json.dumps(data or {}).encode()
        req = urllib.request.Request(url, data=body)
        req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


def send_telegram(report_files):
    report_date = datetime.now().strftime("%d %b %Y")
    idx_by_cat = get_index_categories()

    n50_nav = load_nav(BENCHMARK)
    n50_periods = calc_period_returns(n50_nav) if n50_nav is not None else {}
    n50_1y = n50_periods.get("1Y", 0)

    performers = []
    for cat in CATEGORIES:
        for idx in idx_by_cat.get(cat, []):
            nav = load_nav(idx)
            if nav is not None:
                pr = calc_period_returns(nav)
                own = pr.get("1Y")
                if own is not None:
                    performers.append({"name": idx, "cat": CATEGORY_LABELS[cat], "1y": own, "excess": round(own - n50_1y, 2)})

    performers.sort(key=lambda x: x["excess"], reverse=True)
    outperformers = sum(1 for p in performers if p["excess"] > 0)

    msg = f"📊 *NSE Index Monthly Report*\n📅 {report_date}\n\n"
    msg += f"*NIFTY 50 (1Y):* {n50_1y:+.2f}%\n*Outperformers:* {outperformers}/{len(performers)}\n\n"
    msg += "🏆 *Top 5 (1Y)*\n"
    for i, p in enumerate(performers[:5], 1):
        msg += f"{i}. *{p['name']}* {p['1y']:+.1f}% (_{p['cat']}_)\n"
    msg += "\n📉 *Bottom 5 (1Y)*\n"
    for i, p in enumerate(performers[-5:], 1):
        msg += f"{i}. *{p['name']}* {p['1y']:+.1f}% (_{p['cat']}_)\n"
    msg += f"\n📎 {len(report_files)} reports attached below"

    telegram_api("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    log.info("✓ Telegram summary sent")

    for fp in report_files:
        if not fp.exists():
            continue
        try:
            with open(fp, "rb") as f:
                telegram_api("sendDocument",
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"📊 {fp.name}"},
                    files={"document": (fp.name, f.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
            log.info(f"  ✓ Sent: {fp.name}")
        except Exception as e:
            log.warning(f"  ✗ {fp.name}: {e}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info(f"NSE MONTHLY REPORT — {datetime.now().strftime('%d %b %Y %H:%M')}")
    log.info("=" * 60)

    # Handle --download-only flag for local use
    if "--download-only" in sys.argv:
        log.info("MODE: Download only (run this locally)")
        download_all_data()
        log.info("Done! Now commit and push: git add data/ && git commit -m 'Update data' && git push")
        sys.exit(0)

    start = datetime.now()

    # Step 1: Always try fresh download (new API works without cookies)
    log.info("Step 1/4: Downloading fresh data from NSE...")
    try:
        download_all_data()
    except Exception as e:
        log.warning(f"Download failed: {e}")

    # Check if we have data (either fresh or previously committed)
    csv_count = sum(1 for _ in DATA_DIR.glob("*.csv"))
    if csv_count == 0:
        log.error("No data available. Run locally: python generate_report_ci.py --download-only")
        sys.exit(1)
    log.info(f"  Using {csv_count} CSV files")

    # Step 2: Generate reports
    log.info("Step 2/4: Generating Excel reports...")
    files = generate_reports()
    if not files:
        log.error("No reports generated. Exiting.")
        sys.exit(1)
    log.info(f"Generated {len(files)} reports")

    # Step 3: Email
    if SKIP_EMAIL:
        log.info("Step 3/4: Email SKIPPED")
    elif not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAILS:
        log.info("Step 3/4: Email SKIPPED — no credentials configured")
    else:
        log.info("Step 3/4: Sending email...")
        try:
            send_email(files)
        except Exception as e:
            log.error(f"Email failed: {e}")
            log.error("Reports still saved — check artifacts")

    # Step 4: Telegram
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Step 4/4: Telegram SKIPPED — not configured")
    else:
        log.info("Step 4/4: Sending to Telegram...")
        try:
            send_telegram(files)
        except Exception as e:
            log.error(f"Telegram failed: {e}")

    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"✓ DONE in {elapsed:.0f}s")
