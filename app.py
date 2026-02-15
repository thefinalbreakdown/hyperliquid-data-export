import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Hyperliquid Export", page_icon="📥", layout="centered")

st.title("📥 Hyperliquid Trade History Export")
st.caption("Fetches full perp fill history — chunks by month to avoid rate limits")

HL_URL = "https://api.hyperliquid.xyz/info"

address = st.text_input("Wallet address", placeholder="0x...")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("From", value=datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("To", value=datetime.now())

if st.button("Fetch trades", type="primary"):
    if not address or not address.startswith("0x") or len(address) != 42:
        st.error("Enter a valid 42-character 0x address.")
        st.stop()

    # Build monthly chunks (newest first)
    chunks = []
    chunk_end = end_date
    while chunk_end > start_date:
        chunk_start = max(chunk_end - timedelta(days=30), start_date)
        chunks.append((chunk_start, chunk_end))
        chunk_end = chunk_start - timedelta(days=1)

    all_fills = []
    status = st.empty()
    progress = st.progress(0)

    for ci, (c_start, c_end) in enumerate(chunks):
        s_ms = int(datetime.combine(c_start, datetime.min.time()).timestamp() * 1000)
        e_ms = int(datetime.combine(c_end, datetime.max.time()).timestamp() * 1000)
        current_end = e_ms
        chunk_fills = 0

        for page in range(500):
            status.info(
                f"**{c_start} → {c_end}** (chunk {ci+1}/{len(chunks)}) · "
                f"page {page+1} · {len(all_fills):,} total perp fills"
            )

            resp = None
            for attempt in range(5):
                try:
                    resp = requests.post(HL_URL, json={
                        "type": "userFillsByTime",
                        "user": address,
                        "startTime": s_ms,
                        "endTime": current_end,
                    }, timeout=30)

                    if resp.status_code == 429:
                        wait = 8 * (attempt + 1)
                        status.warning(
                            f"Rate limited — waiting {wait}s "
                            f"(attempt {attempt+1}/5, {len(all_fills):,} fills so far)"
                        )
                        time.sleep(wait)
                        resp = None
                        continue

                    resp.raise_for_status()
                    break
                except Exception as e:
                    if attempt == 4:
                        status.error(f"Failed after 5 attempts: {e}")
                    else:
                        time.sleep(5)
                    resp = None

            if resp is None:
                break

            fills = resp.json()
            if not fills:
                break

            perp_fills = [f for f in fills if not f.get("coin", "").startswith("@")]
            all_fills.extend(perp_fills)
            chunk_fills += len(perp_fills)

            oldest = fills[-1]["time"]
            if oldest >= current_end:
                break
            current_end = oldest - 1

            time.sleep(2.5)

        progress.progress((ci + 1) / len(chunks))

        # Pause between chunks to stay well under rate limits
        if ci < len(chunks) - 1:
            status.info(f"Chunk done ({chunk_fills:,} perp fills). Pausing 5s before next chunk...")
            time.sleep(5)

    progress.empty()
    status.empty()

    if not all_fills:
        st.warning("No fills found.")
        st.stop()

    df = pd.DataFrame(all_fills)
    if "tid" in df.columns:
        df = df.drop_duplicates(subset="tid")
    df = df.sort_values("time").reset_index(drop=True)

    st.success(f"**{len(df):,}** perp fills fetched ({start_date} to {end_date})")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        file_name=f"hl_{address[:10]}_{start_date}_{end_date}.csv",
        mime="text/csv",
        type="primary",
    )
