import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Hyperliquid Export", page_icon="📥", layout="centered")

st.title("📥 Hyperliquid Trade History Export")
st.caption("Fetches full perp fill history via API — no truncation")

HL_URL = "https://api.hyperliquid.xyz/info"

address = st.text_input("Wallet address", placeholder="0x...")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("From", value=datetime.now() - timedelta(days=90))
with col2:
    end_date = st.date_input("To", value=datetime.now())

if st.button("Fetch trades", type="primary"):
    if not address or not address.startswith("0x") or len(address) != 42:
        st.error("Enter a valid 42-character 0x address.")
        st.stop()

    start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    end_ms = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)

    all_fills = []
    current_end = end_ms
    progress = st.empty()
    status = st.empty()

    for page in range(500):
        progress.progress(min(page / 10, 0.99), text=f"Page {page + 1} — {len(all_fills):,} fills so far...")

        for attempt in range(4):
            try:
                resp = requests.post(HL_URL, json={
                    "type": "userFillsByTime",
                    "user": address,
                    "startTime": start_ms,
                    "endTime": current_end,
                }, timeout=15)

                if resp.status_code == 429:
                    wait = 3 * (attempt + 1)
                    status.warning(f"Rate limited — waiting {wait}s (attempt {attempt + 1}/4)...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                fills = resp.json()
                break
            except Exception as e:
                if attempt == 3:
                    status.error(f"Failed after 4 attempts: {e}")
                    fills = None
                    break
                time.sleep(3)
        else:
            fills = None

        if not fills:
            break

        # Filter to perps only (spot coins start with @)
        fills = [f for f in fills if not f.get("coin", "").startswith("@")]

        all_fills.extend(fills)
        oldest = fills[-1]["time"] if fills else current_end
        if oldest >= current_end:
            break
        current_end = oldest - 1
        time.sleep(1.2)

    progress.empty()
    status.empty()

    if not all_fills:
        st.warning("No fills found.")
        st.stop()

    df = pd.DataFrame(all_fills)
    if "tid" in df.columns:
        df = df.drop_duplicates(subset="tid")
    df = df.sort_values("time").reset_index(drop=True)

    st.success(f"**{len(df):,}** fills fetched")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        file_name=f"hl_{address[:10]}_{start_date}_{end_date}.csv",
        mime="text/csv",
        type="primary",
    )
