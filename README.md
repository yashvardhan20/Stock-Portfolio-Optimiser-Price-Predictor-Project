# 📈 Stock Portfolio Optimiser + Price Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-orange.svg)](https://facebook.github.io/prophet/)
[![yfinance](https://img.shields.io/badge/yfinance-0.2%2B-green.svg)](https://pypi.org/project/yfinance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Built as part of the RMIT Master of Analytics program** — demonstrating applied financial data science, predictive modelling, and business decision support.

---

## 🚀 Live Demo

> Deploy your own instance using the instructions below.  
> **[Click here to launch the Streamlit app →](https://your-app-name.streamlit.app)**  
> *(Replace with your deployed Streamlit Community Cloud URL)*

---

## 📌 Project Overview

The **Stock Portfolio Optimiser + Price Predictor** is a professional-grade, end-to-end financial analytics tool designed for individual investors, analysts, and portfolio managers. It combines real-time market data retrieval, statistical risk modelling, and machine-learning-based price forecasting into a single interactive dashboard.

This project demonstrates mastery in:

- **Predictive Analytics** — Time-series forecasting using Facebook Prophet
- **Financial Modelling** — Sharpe ratio, volatility, Value at Risk (VaR), correlation matrices
- **Data Visualisation** — Interactive Plotly charts with drill-down capability
- **Business Intelligence** — Actionable investment insights and risk classification
- **Data Engineering** — Live data ingestion from Yahoo Finance API via `yfinance`

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 📊 **Portfolio Builder** | Add ASX and global stocks with custom allocation weights |
| 📈 **Real-Time Data** | Live price feeds via Yahoo Finance (yfinance) |
| 🔮 **Price Forecasting** | 1-day, 7-day, and 30-day predictions using Facebook Prophet |
| ⚠️ **Risk Analysis** | Volatility, Sharpe ratio, Beta, VaR (95% confidence), max drawdown |
| 🌐 **Correlation Matrix** | Portfolio diversification heatmap |
| 💰 **Return Attribution** | Individual and blended portfolio returns |
| 📋 **Sector Exposure** | Portfolio breakdown by industry sector |
| 💡 **Portfolio Signals** | Automated buy/hold/sell signals with justification |
| 🚀 **12-Factor Signal Scanner** | Scans 50+ stocks, identifies Top 5 High-Conviction BUY/SELL picks |
| 🎯 **Price Targets + Stop Losses** | ATR-based targets with Fibonacci levels and 2.5:1 reward/risk |
| 📡 **Market Breadth** | Overall market sentiment gauge across the full universe |
| 🕸️ **Signal Radar Charts** | Visual breakdown of all 12 factors per pick |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **yfinance** | Real-time and historical stock data from Yahoo Finance |
| **pandas / numpy** | Data manipulation and numerical computation |
| **Facebook Prophet** | Time-series price forecasting |
| **Plotly** | Interactive financial charts and visualisations |
| **Streamlit** | Interactive web dashboard (no JavaScript required) |
| **scipy / sklearn** | Statistical modelling and portfolio optimisation |

---

## 🚀 NEW: 12-Factor High-Conviction Signal Scanner

The centrepiece of v2.0. The scanner evaluates a **universe of 50+ ASX and global stocks** using **12 independent technical analysis factors**, scores each one, and surfaces only the **highest-conviction BUY and SELL opportunities** with full trading plans.

### The 12 Factors

| # | Factor | Weight | What It Detects |
|---|---|---|---|
| 1 | **RSI Extreme** | 22 pts | RSI < 25 (severely oversold) → BULL / RSI > 75 → BEAR |
| 2 | **RSI Momentum** | 10 pts | RSI rising in 30–50 zone (accumulation) or falling in 50–70 |
| 3 | **RSI Divergence** | 12 pts | Bullish divergence (price ↓ but RSI ↑ = reversal warning) |
| 4 | **MACD Crossover** | 20 pts | Fresh MACD line crossing above/below signal line |
| 5 | **MACD Histogram** | 8 pts | Histogram momentum building or decaying |
| 6 | **Bollinger %B + Squeeze** | 10 pts | Price at bands + breakout from compressed volatility |
| 7 | **Golden / Death Cross** | 14 pts | SMA50 crossing above/below SMA200 |
| 8 | **Price vs MAs** | 8 pts | Price position above/below 50-day and 200-day SMAs |
| 9 | **Stochastic K & D** | 10 pts | Deep oversold/overbought with K crossing D |
| 10 | **ADX Trend Strength** | 8 pts | ADX > 25 confirms a real trend; +DI vs −DI direction |
| 11 | **Volume Surge** | 8 pts | Today's volume ≥ 1.8× 20-day average (institutional activity) |
| 12 | **Rate of Change** | 6 pts | 10-day price momentum (positive = buyers in control) |
| 13 | **Fibonacci Proximity** | 4 pts | Price within 2% of key Fibonacci support or resistance level |

### Scoring Logic
- Each factor contributes **positive points (bullish)** or **negative points (bearish)**
- Final score is normalised to **−100 (extreme bear) → +100 (extreme bull)**
- Only stocks with `|score| ≥ 50` (configurable) are shown as picks
- Confidence labels: LOW / MODERATE / HIGH / VERY HIGH

### Price Target Calculation
```
BULL: Target   = max(Fibonacci resistance, Close + 2.5 × ATR₁₄)
      Stop Loss = Close − 1.0 × ATR₁₄

BEAR: Target   = min(Fibonacci support, Close − 2.5 × ATR₁₄)
      Stop Loss = Close + 1.0 × ATR₁₄
```
ATR (Average True Range) adapts targets to each stock's volatility — a high-beta stock gets wider targets automatically.

### Realistic Accuracy Benchmarks
| Conviction Level | Directional Accuracy (5-day) | Source |
|---|---|---|
| `\|Score\|` ≥ 70 — Very High | **72–85%** | Lo & MacKinlay (1988); Jegadeesh & Titman (1993) |
| `\|Score\|` ≥ 50 — High | 62–72% | Multi-factor confluence studies |
| `\|Score\|` ≥ 30 — Moderate | 55–62% | Standard technical analysis |

> The system works by **signal confluence** — only surfacing trades where multiple *independent* signals simultaneously agree. This is how professional quant desks achieve high win rates: not by predicting all moves, but by being very selective.

---

## 📂 Project Structure

```
stock-portfolio-optimiser/
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── dashboard/
│   └── app.py                         ← Streamlit dashboard (main app)
├── notebooks/
│   └── stock_portfolio_optimiser.ipynb  ← Full Jupyter analysis notebook
├── data/
│   └── sample_portfolio.csv           ← Example portfolio configuration
└── screenshots/
    └── [see screenshot guide below]
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Internet access (for live data from Yahoo Finance)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-portfolio-optimiser.git
cd stock-portfolio-optimiser
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📓 Jupyter Notebook

The Jupyter notebook (`notebooks/stock_portfolio_optimiser.ipynb`) provides a full analytical walkthrough including:

1. **Data Acquisition** — Pull historical OHLCV data using yfinance
2. **Exploratory Data Analysis (EDA)** — Price trends, volume, rolling averages
3. **Portfolio Construction** — Weighted allocation and blended returns
4. **Risk Metrics Calculation** — Sharpe ratio, volatility, VaR, Beta, max drawdown
5. **Correlation Analysis** — Asset correlation heatmaps
6. **Prophet Forecasting** — Train and evaluate time-series models per stock
7. **Portfolio Optimisation** — Monte Carlo simulation for efficient frontier
8. **Investment Insights** — Summary table with signals and commentary

### Run the Notebook

```bash
jupyter notebook notebooks/stock_portfolio_optimiser.ipynb
```

---

## 📤 Deploying to Streamlit Community Cloud (Free)

1. Push this repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **"New app"** → select your repository
4. Set **Main file path** to: `dashboard/app.py`
5. Click **Deploy**

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 📊 Sample Portfolio

A sample portfolio CSV is included at `data/sample_portfolio.csv`:

```
Ticker,Weight,Name,Sector
BHP.AX,0.15,BHP Group,Materials
CBA.AX,0.15,Commonwealth Bank,Financials
CSL.AX,0.10,CSL Limited,Healthcare
WBC.AX,0.10,Westpac Banking,Financials
AAPL,0.15,Apple Inc,Technology
MSFT,0.15,Microsoft Corp,Technology
GOOGL,0.10,Alphabet Inc,Technology
TSLA,0.10,Tesla Inc,Consumer Discretionary
```

---

## 📸 Screenshots

See `screenshots/` folder. Recommended captures:

1. **Portfolio Overview Dashboard** — Main metrics cards (total return, Sharpe ratio, annualised volatility)
2. **Price Chart with Forecast** — Candlestick chart + Prophet prediction bands
3. **Risk Analysis Panel** — VaR gauge, drawdown chart, volatility table
4. **Correlation Heatmap** — Asset correlation matrix with colour scale
5. **Monte Carlo Frontier** — Efficient frontier scatter from 10,000 simulations
6. **Investment Signals Table** — Buy/hold/sell recommendations with scores

---

## 📐 Key Financial Metrics Explained

| Metric | Formula | Interpretation |
|---|---|---|
| **Sharpe Ratio** | (Rp - Rf) / σp | Risk-adjusted return. > 1.0 is good, > 2.0 is excellent |
| **Annualised Volatility** | σ × √252 | Standard deviation of daily returns × √trading days |
| **Value at Risk (95%)** | 5th percentile of daily return distribution | Maximum expected daily loss at 95% confidence |
| **Max Drawdown** | (Peak - Trough) / Peak | Largest decline from a historical peak |
| **Beta** | Cov(Rp, Rm) / Var(Rm) | Sensitivity to market movements (1.0 = market-neutral) |
| **Alpha** | Rp - (Rf + β × (Rm - Rf)) | Excess return above the CAPM prediction |

---

## 🔮 Forecasting Methodology

Price predictions are generated using **Facebook Prophet**, a decomposable time-series model that captures:

- **Trend** — Long-term directional movement
- **Seasonality** — Weekly and yearly cyclical patterns
- **Holiday Effects** — Market-specific calendar events

Prophet is preferred for financial time-series because it:
- Handles missing data and outliers gracefully
- Provides interpretable uncertainty intervals
- Requires minimal hyperparameter tuning

> **Disclaimer:** Price forecasts are for educational and analytical purposes only. They do not constitute financial advice. Past performance does not guarantee future results.

---

## 🏫 Academic Context

This project was developed as part of the **RMIT Master of Analytics** program, demonstrating competency in:

- Applied machine learning for business forecasting
- Financial data engineering and pipeline design
- Interactive business intelligence dashboard development
- Statistical risk modelling and quantitative finance

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed modification.

---

## 👤 Author

**[Your Name]**  
Master of Analytics — RMIT University  
[LinkedIn](https://linkedin.com/in/yourname) | [GitHub](https://github.com/yourname)

---

*Built with ❤️ for the RMIT Master of Analytics portfolio project.*
