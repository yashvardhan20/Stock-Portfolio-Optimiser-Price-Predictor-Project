# 📸 Screenshot Guide — Stock Portfolio Optimiser

Below are the 6 recommended screenshots to capture for your GitHub README and portfolio.

---

## Screenshot 1: Portfolio Overview Dashboard
**Filename:** `01_portfolio_overview.png`

**What to capture:**
- Run the app with the default 8-stock portfolio (BHP.AX, CBA.AX, CSL.AX, AAPL, MSFT, GOOGL, TSLA, JPM)
- Make sure you are on the **"Portfolio Overview"** tab
- Capture the full page including:
  - The 5 KPI cards at the top (Total Return, Annualised Return, Sharpe Ratio, Volatility, Max Drawdown)
  - The cumulative returns line chart showing all stocks + blended portfolio
  - The holdings summary table below

**Pro tip:** Set the period to "1y" and zoom in the browser to 90% to get everything in one screenshot.

---

## Screenshot 2: Price Forecast Chart
**Filename:** `02_price_forecast.png`

**What to capture:**
- Select **"Price Forecasting"** tab
- Set the **Detail Ticker** to `AAPL` (in the sidebar)
- Set **Forecast Horizon** to `30 Days`
- Click **Run Analysis**
- Capture the candlestick chart showing:
  - Historical OHLC candles in green/red
  - The blue dashed forecast line extending into the future
  - The shaded confidence interval (95% CI)
  - The 4 metric cards below (Current Price, Predicted Price, Lower/Upper Bounds)

---

## Screenshot 3: Risk Analysis Panel
**Filename:** `03_risk_analysis.png`

**What to capture:**
- Select **"Risk Analysis"** tab
- Capture the full view including:
  - Return distribution histogram with VaR 95% line marked in red
  - Drawdown chart showing the depth and recovery of drawdowns
  - Risk metrics comparison table (all stocks side by side)
- Optionally scroll to show the portfolio-level risk summary box

---

## Screenshot 4: Correlation Heatmap
**Filename:** `04_correlation_heatmap.png`

**What to capture:**
- Select **"Correlation & Sector"** tab
- Capture:
  - The full correlation heatmap with colour scale (red = negative, blue = positive)
  - The correlation alert boxes below (if any highly correlated pairs exist)
  - The rolling 90-day correlation chart (set to `AAPL` vs `MSFT` for a nice chart)

---

## Screenshot 5: Efficient Frontier
**Filename:** `05_efficient_frontier.png`

**What to capture:**
- Select **"Efficient Frontier"** tab
- Wait for the Monte Carlo simulation to complete
- Capture:
  - The scatter plot showing 5,000 simulated portfolios coloured by Sharpe ratio
  - The red star marking YOUR current portfolio position
  - The two metric cards (Max Sharpe Portfolio, Min Volatility Portfolio)

---

## Screenshot 6: Investment Signals
**Filename:** `06_investment_signals.png`

**What to capture:**
- Select **"Investment Signals"** tab
- Capture:
  - The coloured signal boxes for each stock (green = BUY, yellow = HOLD, red = SELL)
  - The signals summary table
  - The 3 KPI cards at the bottom (BUY / HOLD / SELL count)
  - The portfolio-level recommendation insight box

---

## How to Take High-Quality Screenshots

### Option A: Browser Built-in
1. Open Chrome or Firefox with the Streamlit app
2. Press `F12` → Device toolbar → Set width to 1440px
3. Use `Ctrl+Shift+S` (Windows) or `Cmd+Shift+4` (Mac) to capture

### Option B: Full-Page Screenshot Tool
- Install the Chrome extension **"GoFullPage"** for complete page captures
- This is especially useful for the holdings table and risk metrics table

### Option C: Streamlit Screenshot
- In the Streamlit app, use the camera icon (⋮ menu → "Download screenshot") if available

---

## Recommended Image Sizes
- **Width:** 1440px (wide monitor resolution)
- **Height:** As tall as needed to capture all content
- **Format:** PNG (preferred) or JPG at 90% quality
- **Tool:** Use [Squoosh](https://squoosh.app/) to compress before uploading to GitHub

---

*After capturing, place all screenshots in this `screenshots/` folder and update the README.md with actual image links.*
