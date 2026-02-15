# Hyperliquid Wallet Analyzer

Analyze any Hyperliquid wallet's perp trading performance and evaluate copy trade feasibility.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red)

## Features

- **Full trade history** — Pulls complete fill history via Hyperliquid's API (no more truncated CSV exports)
- **PnL breakdown** — Realized PnL, fees, win rate, profit factor, Sharpe ratio, max drawdown
- **Per-token analysis** — See which tokens are driving profits/losses and how concentrated the PnL is
- **Copy trade feasibility** — Checks if your account size can realistically replicate the trader's positions (minimum order sizes, scaling ratios, estimated fees)
- **Open positions** — View the wallet's current exposure, unrealized PnL, and leverage
- **CSV upload support** — Also works with Hyperliquid's UI CSV exports (with FIFO PnL reconstruction)
- **Exportable data** — Download the full processed dataset as CSV

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/hyperliquid-analyzer.git
cd hyperliquid-analyzer

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Usage

### Option 1: Fetch from API (recommended)

1. Paste any Hyperliquid wallet address (0x...) in the sidebar
2. Set the date range
3. Click **Analyze wallet**

This pulls the complete trade history directly from Hyperliquid's public API. No API key needed — all data is on-chain.

### Option 2: Upload CSV

1. Export a CSV from the Hyperliquid UI (Trade History → Export as CSV)
2. Upload it via the sidebar

Note: CSV exports from the UI are often truncated. The API method gives you the full history.

### Copy Trade Analysis

1. Enter your account size (e.g. $1,000)
2. Estimate the trader's equity (check their max exposure ÷ likely leverage)
3. The tool checks what percentage of trades would fall below Hyperliquid's minimum order size and suggests a minimum viable account

## How it works

- **API data**: Uses `userFillsByTime` endpoint with automatic pagination to fetch all fills
- **PnL calculation**: For API data, uses Hyperliquid's native `closedPnl` field. For CSV uploads, reconstructs PnL using FIFO matching of opens to closes
- **Open positions**: Fetches live data from `clearinghouseState` endpoint
- **Copy feasibility**: Scales all trade notionals by your account ratio and checks against $10 minimum order size

## Project Structure

```
hyperliquid-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.9+
- Internet connection (to reach Hyperliquid API)

## Disclaimer

This tool is for informational purposes only. Past performance does not guarantee future results. Copy trading involves significant risk of loss. Always do your own research and never risk more than you can afford to lose.
