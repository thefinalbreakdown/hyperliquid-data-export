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

    MAX_FILLS = 10000

    all_fills = []
    current_end = end_ms
    status = st.empty()

    for page in range(100):
        status.info(f"Page {page + 1} — {len(all_fills):,} perp fills so far...")

        for attempt in range(3):
            try:
                resp = requests.post(HL_URL, json={
                    "type": "userFillsByTime",
                    "user": address,
                    "startTime": start_ms,
                    "endTime": current_end,
                }, timeout=30)

                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    status.warning(f"Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                fills = resp.json()
                break
            except Exception as e:
                if attempt == 2:
                    status.error(f"Stopped: {e}")
                    fills = None
                time.sleep(5)
        else:
            fills = None

        if not fills:
            break

        # Filter perps only (spot coins start with @)
        perp_fills = [f for f in fills if not f.get("coin", "").startswith("@")]
        all_fills.extend(perp_fills)

        oldest = fills[-1]["time"]
        if oldest >= current_end:
            break
        current_end = oldest - 1

        if len(all_fills) >= MAX_FILLS:
            status.warning(f"Hit {MAX_FILLS:,} fill cap — stopping. Adjust date range for more.")
            break

        time.sleep(2)

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
