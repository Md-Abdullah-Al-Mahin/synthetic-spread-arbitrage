# synthetic-spread-arbitrage

Project Structure
synthetic-optimizer/
│
├── data/
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Cleaned data
│   └── results/                # Model outputs
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        # Data collection & cleaning
│   ├── synthetic_pricer.py     # Synthetic vs cash economics
│   ├── statistical_models.py   # Regression, time series, etc.
│   ├── risk_analytics.py       # VaR, Monte Carlo, Greeks
│   ├── backtester.py           # Strategy backtesting
│   └── visualizer.py           # Charts and dashboards
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_results_presentation.ipynb
│
├── tests/
│   └── test_models.py
│
├── config.py                   # Configuration parameters
├── main.py                     # Main execution script
├── requirements.txt
└── README.md


# **Detailed Component Breakdown**

Let me break down each module of your project with examples, formulas, and step-by-step explanations.

---

## **COMPONENT 1: Data Pipeline** (`data_pipeline.py`)

### **What It Does**
Downloads and processes raw market data into usable formats for analysis.

### **Key Functions Explained**

#### **1.1: `fetch_market_data()`**

**Purpose:** Download historical stock prices

**Input:**
- List of tickers: `['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']`
- Date range: `2023-01-01` to `2025-12-31`

**What It Returns:**
```python
# DataFrame with dates as rows, tickers as columns
         AAPL    MSFT    GOOGL   TSLA    NVDA
Date                                          
2023-01-03  125.07  239.58  88.37  108.10  146.14
2023-01-04  126.36  240.35  89.50  113.64  149.43
2023-01-05  125.02  237.49  88.93  110.34  146.84
...
```

**Why You Need It:** Everything else depends on having price history.

---

#### **1.2: `calculate_returns()`**

**Purpose:** Convert prices to percentage returns

**Formula:**
```
Return_t = (Price_t - Price_(t-1)) / Price_(t-1)
```

**Example Calculation:**
```python
AAPL on Jan 3: $125.07
AAPL on Jan 4: $126.36

Return = (126.36 - 125.07) / 125.07 = 0.0103 = 1.03%
```

**Output:**
```python
         AAPL      MSFT     GOOGL    TSLA     NVDA
Date                                              
2023-01-04  0.0103   0.0032   0.0128   0.0512   0.0225
2023-01-05 -0.0106  -0.0120  -0.0064  -0.0291  -0.0173
...
```

**Why You Need It:** Returns are standardized, easier to compare across stocks and calculate volatility.

---

#### **1.3: `calculate_realized_volatility()`**

**Purpose:** Measure how much a stock's price fluctuates

**Formula:**
```
Volatility = StdDev(Returns_last_30_days) × √252
```

**Step-by-Step Example:**

1. Get last 30 days of returns for AAPL:
   ```
   [-0.5%, 1.2%, -0.8%, 2.1%, 0.3%, ..., 1.5%]
   ```

2. Calculate standard deviation:
   ```
   Daily Std Dev = 1.5%
   ```

3. Annualize (multiply by √252):
   ```
   Annual Volatility = 1.5% × √252 = 1.5% × 15.87 = 23.8%
   ```

**Output:**
```python
         AAPL   MSFT   GOOGL   TSLA   NVDA
Date                                      
2023-02-15  0.238  0.215  0.264  0.487  0.298
2023-02-16  0.241  0.218  0.261  0.492  0.301
...
```

**Why You Need It:** 
- Volatility is the PRIMARY driver of financing spreads
- Higher vol = higher spreads = more expensive synthetics

---

#### **1.4: `get_dividend_data()`**

**Purpose:** Get dividend payment history

**Output:**
```python
             AAPL   MSFT   GOOGL
Date                            
2023-02-10   0.23   0.68   0.00
2023-05-12   0.24   0.68   0.00
2023-08-11   0.24   0.68   0.00
2023-11-10   0.24   0.75   0.00
```

**Why You Need It:** 
- Synthetics have different dividend treatment than cash
- Affects total cost comparison

---

### **What This Component Produces**

**Three Core Datasets:**
1. **Price DataFrame** - raw prices
2. **Returns DataFrame** - daily % changes
3. **Volatility DataFrame** - 30-day rolling vol

**These feed into ALL other components.**

---

## **COMPONENT 2: Synthetic Pricer** (`synthetic_pricer.py`)

### **What It Does**
Calculates the all-in cost of synthetic positions vs. cash positions.

### **Key Functions Explained**

#### **2.1: `calculate_financing_cost()`**

**Purpose:** Calculate what the prime broker charges for synthetic exposure

**Formula:**
```
Financing Cost = (SOFR + Spread) × Notional × (Days/360)
```

**Example Calculation:**

**Scenario:** You want synthetic exposure to 1,000 shares of AAPL at $150/share for 90 days

```python
Inputs:
- SOFR rate: 4.5% (0.045)
- Spread: 50 bps (0.005)
- Notional: 1,000 shares × $150 = $150,000
- Days: 90

Calculation:
Financing Cost = (0.045 + 0.005) × $150,000 × (90/360)
              = 0.05 × $150,000 × 0.25
              = $1,875

Daily Cost = $1,875 / 90 = $20.83 per day
```

**Output:**
```python
{
    'total_cost': 1875.00,
    'daily_cost': 20.83,
    'annual_rate': 0.05  # 5%
}
```

**Why You Need It:** This is the BASE cost of holding a synthetic position.

---

#### **2.2: `estimate_spread_from_volatility()`**

**Purpose:** Model how spreads change with market conditions

**The Key Insight:** Prime brokers charge more when stocks are volatile (more risk for them)

**Formula (Simple Linear Model):**
```
Spread = Base_Spread + (Vol_Coefficient × Volatility)
```

**Example Calculation:**

```python
# Calibrated parameters (you'd estimate these via regression)
Base_Spread = 15 bps (0.0015)
Vol_Coefficient = 20 bps per vol point (0.002)

# For AAPL with 25% volatility:
Estimated_Spread = 0.0015 + (0.002 × 0.25)
                 = 0.0015 + 0.0005
                 = 0.0020 (20 bps)

# For TSLA with 50% volatility:
Estimated_Spread = 0.0015 + (0.002 × 0.50)
                 = 0.0015 + 0.0010
                 = 0.0025 (25 bps)
```

**Real-World Interpretation:**
- AAPL (low vol): Spread = 20 bps → Annual cost = SOFR + 0.20% = 4.7%
- TSLA (high vol): Spread = 25 bps → Annual cost = SOFR + 0.25% = 4.75%

**Why You Need It:** Without access to real spread data, this models how spreads behave.

---

#### **2.3: `calculate_dividend_impact()`**

**Purpose:** Account for different dividend treatment

**The Concept:**
- **Cash Position:** You receive dividends, but might pay tax
- **Synthetic (TRS):** Often structured as "gross dividend" (no withholding)

**Formula:**
```
Cash Dividend = Dividend × (1 - Tax_Rate)
Synthetic Dividend = Dividend × 1.0  (gross treatment)
Advantage = Synthetic - Cash
```

**Example:**

```python
Stock: MSFT
Annual Dividend Yield: 2.5% (0.025)
Tax Rate: 30% (0.30)

Cash Position:
- You receive: 2.5% × (1 - 0.30) = 1.75%
- After-tax yield: 1.75%

Synthetic Position:
- You receive: 2.5% (gross)
- Effective yield: 2.5%

Advantage = 2.5% - 1.75% = 0.75%

On $100,000 position: 0.75% = $750 annual advantage
```

**Why You Need It:** This can offset synthetic financing costs.

---

#### **2.4: `calculate_total_cost_of_carry()`**

**Purpose:** **THE MAIN CALCULATION** - compare all-in costs

**Full Formula:**
```
Net Synthetic Cost = Financing Cost - Dividend Advantage + Transaction Costs
Net Cash Cost = Borrow Cost (if short) OR Opportunity Cost (if long) + Transaction Costs
```

**Complete Example:**

**Scenario:** Long position in MSFT, $100,000 notional, 90 days

```python
SYNTHETIC ROUTE:
  Notional: $100,000
  SOFR: 4.5%
  Volatility: 22%
  Estimated Spread: 15 + (20 × 0.22) = 19.4 bps
  Total Rate: 4.5% + 0.194% = 4.694%
  
  Financing Cost = $100,000 × 0.04694 × (90/360) = $1,173.50
  
  Dividend Yield: 2.5%
  Div Treatment Advantage: 0.75% (from above)
  Dividend Benefit = $100,000 × 0.0075 × (90/360) = $187.50
  
  Net Cost = $1,173.50 - $187.50 = $986.00

CASH ROUTE:
  Opportunity Cost = $100,000 × 0.045 × (90/360) = $1,125.00
  Transaction Costs = $25
  Net Cost = $1,125.00 + $25 = $1,150.00

DECISION:
  Synthetic Cost: $986
  Cash Cost: $1,150
  Savings with Synthetic: $164 (14.3% cheaper)
  
  → RECOMMENDATION: Use synthetic position
```

**Output:**
```python
{
    'synthetic_cost': 986.00,
    'cash_cost': 1150.00,
    'savings': 164.00,
    'synthetic_rate': 0.04694,
    'recommendation': 'SYNTHETIC'
}
```

**Why You Need It:** This is the DECISION ENGINE that tells you which route is cheaper.

---

#### **2.5: `calculate_basis()`**

**Purpose:** Track the spread differential between synthetic and cash

**Formula:**
```
Basis = Synthetic Financing Rate - Cash Equivalent Rate
```

**Example:**
```python
Synthetic Rate: 4.7% (SOFR + spread)
Cash Borrow Rate: 3.2%

Basis = 4.7% - 3.2% = 1.5% = 150 bps

Interpretation: Synthetics are 150 bps MORE expensive than cash
```

**Why This Matters:** The basis fluctuates over time - that's where the trading opportunity is!

---

### **What This Component Produces**

**For Each Stock, Each Day:**
```python
{
    'ticker': 'AAPL',
    'date': '2024-06-15',
    'synthetic_cost': 1250.00,
    'cash_cost': 1400.00,
    'basis': 0.015,  # 150 bps
    'recommendation': 'SYNTHETIC',
    'estimated_spread': 0.0019
}
```

**This feeds into the statistical models to find patterns.**

---

## **COMPONENT 3: Statistical Models** (`statistical_models.py`)

### **What It Does**
Finds patterns in the data and predicts future behavior.

### **Key Functions Explained**

#### **3.1: `regression_spread_drivers()`**

**Purpose:** Understand WHAT drives spread changes

**The Question:** Why do spreads widen or tighten?

**Multiple Regression Formula:**
```
Spread = β₀ + β₁(Volatility) + β₂(Liquidity) + β₃(VIX) + ε
```

**Step-by-Step Example:**

**Step 1: Collect Data**
```python
Date        Spread   Vol    Liquidity  VIX
2024-01-03  0.0018   0.23   0.0005    14.2
2024-01-04  0.0019   0.24   0.0006    15.1
2024-01-05  0.0021   0.27   0.0007    16.8
...
(500 observations)
```

**Step 2: Run Regression**
```python
from sklearn.linear_model import LinearRegression

X = features[['Vol', 'Liquidity', 'VIX']]
y = spreads

model = LinearRegression()
model.fit(X, y)
```

**Step 3: Get Results**
```python
Results:
  R² = 0.67  (model explains 67% of spread variation)
  
  Coefficients:
    Intercept (β₀) = 0.0008  (80 bps base)
    Volatility (β₁) = 0.0020  (20 bps per vol point)
    Liquidity (β₂) = 1.5000  (150 bps per 1% bid-ask)
    VIX (β₃) = 0.0001  (1 bp per VIX point)
  
  P-values:
    Volatility: p < 0.001  (highly significant)
    Liquidity: p = 0.023   (significant)
    VIX: p = 0.089         (marginally significant)
```

**Step 4: Interpret**

**Example Prediction:**
```python
New Market Conditions:
- Vol = 30%
- Liquidity (bid-ask) = 0.08%
- VIX = 18

Predicted Spread = 0.0008 + (0.0020 × 0.30) + (1.5 × 0.0008) + (0.0001 × 18)
                 = 0.0008 + 0.0006 + 0.0012 + 0.0018
                 = 0.0044 (44 bps)
```

**Why You Need It:** 
- Understand spread drivers
- Predict future spreads
- Identify when spreads are "too high" or "too low"

---

#### **3.2: `fit_garch_volatility()`**

**Purpose:** Forecast future volatility to predict future spreads

**The Concept:** Volatility is NOT random - it clusters and mean-reverts

**GARCH(1,1) Formula:**
```
σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁

Where:
- σ²ₜ = Variance (vol squared) today
- r²ₜ₋₁ = Yesterday's squared return
- σ²ₜ₋₁ = Yesterday's variance
- ω = Long-run variance
- α = Weight on recent shocks
- β = Weight on historical variance
```

**Example:**

**Step 1: Fit Model to Historical Returns**
```python
from arch import arch_model

returns = stock_returns['AAPL'] * 100  # Convert to percentage
model = arch_model(returns, vol='Garch', p=1, q=1)
fitted = model.fit()

Parameters Estimated:
  ω = 0.02
  α = 0.08  (recent shocks matter)
  β = 0.90  (persistence is high)
```

**Step 2: Forecast Volatility**
```python
Current volatility: 25%
Forecast 10 days ahead:

Day 1: 25.3%
Day 2: 25.8%
Day 3: 26.4%
Day 4: 27.1%
Day 5: 27.8%
...
Day 10: 29.2%
```

**Step 3: Predict Spread Change**
```python
Current spread (at 25% vol): 20 bps
Predicted spread (at 29.2% vol): 20 + (29.2 - 25) × 2 = 28.4 bps

Expected Spread Widening: 8.4 bps over next 10 days
```

**Trading Signal:**
```
If you need to enter a large synthetic position:
- Lock in financing NOW at 20 bps
- Avoid waiting 10 days when spread might be 28 bps
- Potential savings: 8.4 bps annually
```

**Why You Need It:** Predict spread changes BEFORE they happen.

---

#### **3.3: `calculate_zscore()`**

**Purpose:** Identify when spreads are abnormally high/low (TRADING SIGNALS)

**Formula:**
```
Z-Score = (Current Value - Rolling Mean) / Rolling Std Dev
```

**Example:**

**Step 1: Calculate Historical Statistics**
```python
AAPL Synthetic Basis (last 252 trading days):
  Mean = 45 bps (0.0045)
  Std Dev = 15 bps (0.0015)
```

**Step 2: Calculate Today's Z-Score**
```python
Today's Basis = 75 bps (0.0075)

Z = (0.0075 - 0.0045) / 0.0015
  = 0.0030 / 0.0015
  = 2.0
```

**Step 3: Interpret**
```
Z-Score = +2.0

Meaning: Today's spread is 2 standard deviations ABOVE average

Historical Context:
- This happens only ~2.5% of the time
- Spread is "too wide"
- Mean reversion likely

Trading Signal: AVOID synthetics, use cash instead
```

**Different Scenarios:**

| Z-Score | Interpretation | Signal | Action |
|---------|---------------|--------|---------|
| +2.5 | 2.5σ above mean | STRONG SELL | Definitely use cash |
| +1.0 | Slightly elevated | WEAK SELL | Slight preference for cash |
| 0.0 | At average | NEUTRAL | No preference |
| -1.5 | Below average | WEAK BUY | Slight preference for synthetic |
| -2.3 | 2.3σ below mean | STRONG BUY | Definitely use synthetic |

**Complete Trading Strategy:**
```python
if zscore > 2.0:
    signal = "SHORT_BASIS"  # Spread too wide, use cash
    expected_return = "Wait for reversion to mean (+45 bps profit)"
    
elif zscore < -2.0:
    signal = "LONG_BASIS"  # Spread too tight, use synthetic
    expected_return = "Wait for reversion to mean (+45 bps profit)"
    
elif abs(zscore) < 0.5:
    signal = "EXIT"  # Mean reversion complete
    action = "Close position, take profits"
```

**Why You Need It:** This is your PRIMARY trading signal generator.

---

#### **3.4: `arima_forecast()`**

**Purpose:** Time series forecasting for spread trends

**ARIMA Model:** Autoregressive Integrated Moving Average

**Formula:**
```
ARIMA(p, d, q):
- p = autoregressive terms (past values)
- d = differencing (make series stationary)
- q = moving average terms (past errors)
```

**Example:**

```python
Historical Spread Data:
[45, 47, 46, 48, 52, 54, 53, 55, 58, 60] bps

ARIMA(1,1,1) Model:
  Forecast next 5 days:
  Day 1: 62 bps
  Day 2: 63 bps
  Day 3: 64 bps
  Day 4: 64 bps
  Day 5: 65 bps

Interpretation: Upward trend continues → spreads widening
Action: Lock in financing before further widening
```

**Why You Need It:** Complements z-score by capturing trends.

---

#### **3.5: `hypothesis_test_cost_savings()`**

**Purpose:** PROVE your strategy works statistically

**The Question:** "Is the cost savings from optimal synthetic selection REAL, or just luck?"

**Hypothesis Test:**
```
H₀ (Null): Mean savings ≤ 0 (no benefit)
H₁ (Alternative): Mean savings > 0 (real benefit)
```

**Example:**

**Step 1: Collect Savings Data**
```python
30 trades over backtest period:
Savings per trade (in bps):
[12, -3, 18, 22, 15, 8, 25, -2, 19, 14, 16, 20, 9, 11, 23, ...]

Mean Savings = 14.2 bps
Std Dev = 7.8 bps
N = 30 trades
```

**Step 2: Calculate T-Statistic**
```python
t = (Mean - 0) / (Std Dev / √n)
t = (14.2 - 0) / (7.8 / √30)
t = 14.2 / 1.42
t = 10.0
```

**Step 3: Get P-Value**
```python
Degrees of freedom = 30 - 1 = 29
t = 10.0

P-value < 0.0001  (extremely significant)
```

**Step 4: Conclusion**
```python
Result: REJECT null hypothesis

Statistical Interpretation:
- Probability this result is due to chance: < 0.01%
- 95% Confidence Interval: [11.3 bps, 17.1 bps]
- Conclusion: Strategy delivers REAL, statistically significant savings

Interview Talking Point:
"The strategy generated 14.2 bps average savings per trade with 
statistical significance (t=10.0, p<0.0001), confirming this is not 
due to random chance."
```

**Why You Need It:** Adds academic rigor, proves you're not data mining.

---

### **What This Component Produces**

**Regression Model Output:**
```python
{
    'r_squared': 0.67,
    'coefficients': {
        'volatility': 0.0020,
        'liquidity': 1.5000,
        'vix': 0.0001
    },
    'model': <trained_model>
}
```

**Z-Score Series:**
```python
Date        Z-Score  Signal
2024-06-10   -0.5    NEUTRAL
2024-06-11   -1.2    WEAK_BUY
2024-06-12   -2.3    STRONG_BUY ← Enter synthetic
2024-06-13   -1.8    HOLD
2024-06-14   -0.3    EXIT ← Take profit
```

**GARCH Forecast:**
```python
{
    'current_vol': 0.25,
    'forecast_10day': 0.292,
    'expected_spread_change': 0.0084
}
```

---

## **COMPONENT 4: Risk Analytics** (`risk_analytics.py`)

### **What It Does**
Quantifies risks and stress-tests your strategy.

### **Key Functions Explained**

#### **4.1: `calculate_var()` - Value at Risk**

**Purpose:** "What's the worst loss I can expect 95% of the time?"

**Formula:**
```
VaR(95%) = 5th percentile of return distribution
```

**Example:**

**Step 1: Historical Returns**
```python
Your strategy's daily P&L over 252 days (sorted):
[-$8,500, -$6,200, -$5,100, -$3,800, ..., $2,100, $3,500, $4,200]
```

**Step 2: Find 5th Percentile**
```python
5% of 252 days = 12.6 days → 13th worst day
13th worst loss = -$5,100

VaR(95%) = $5,100
```

**Step 3: Interpret**
```
Interpretation:
"On 95% of days, my loss will not exceed $5,100"
OR
"I expect to lose more than $5,100 only 5% of the time (13 days/year)"

Risk Management Application:
- Set position limits to ensure VaR doesn't exceed risk tolerance
- If max acceptable loss is $10,000, current strategy is safe
```

**Why You Need It:** Standard risk measure used by all trading desks.

---

#### **4.2: `calculate_cvar()` - Conditional VaR**

**Purpose:** "When things go REALLY bad, how bad?"

**Formula:**
```
CVaR(95%) = Average of all losses beyond VaR threshold
```

**Example:**

**Using same data:**
```python
VaR(95%) = -$5,100

Losses worse than VaR (worst 13 days):
[-$8,500, -$7,200, -$6,800, -$6,500, -$6,200, -$6,000, 
 -$5,900, -$5,700, -$5,500, -$5,400, -$5,300, -$5,200, -$5,100]

CVaR = Average of these = -$6,331

Interpretation:
"When I'm in the worst 5% of days, average loss is $6,331"
```

**Comparison:**
```
VaR(95%): $5,100 ← threshold
CVaR(95%): $6,331 ← average beyond threshold

Gap = $1,231 ← shows tail risk
```

**Why You Need It:** Shows if you have "fat tails" (extreme events).

---

#### **4.3: `monte_carlo_simulation()`**

**Purpose:** Simulate thousands of possible futures

**The Process:**

**Step 1: Define Parameters**
```python
Initial Portfolio: $1,000,000
Expected Return (drift): 8% annually
Volatility: 25% annually
Simulation Horizon: 252 days (1 year)
Number of Simulations: 10,000
```

**Step 2: Simulate Price Paths**

For each of 10,000 simulations:
```python
Day 0: $1,000,000

For each day:
  Random shock ~ N(0,1)
  
  Return_t = (0.08/252) + (0.25/√252) × shock
  
  Value_t = Value_(t-1) × (1 + Return_t)

End with final value after 252 days
```

**Step 3: Results**
```python
10,000 Final Values:
[$847,000, $923,000, $1,050,000, $1,205,000, ..., $1,380,000]

Statistics:
  Mean Final Value: $1,083,000
  Median: $1,071,000
  Std Dev: $187,000
  
  5th Percentile (VaR): $781,000 ← worst 5%
  95th Percentile: $1,432,000 ← best 5%
  
  Probability of Loss: 23.4%
  Probability of > 10% Return: 67.2%
```

**Step 4: Visualize Distribution**
```
Distribution of Final Values:

      |
 800  |  *
1000  |  ****
1200  |  **********  ← Most likely outcomes
1400  |  ****
1600  |  *
      |
     
VaR: Only 5% of scenarios below $781k
```

**Why You Need It:** 
- Understand full range of possible outcomes
- Quantify probability of hitting targets
- Stress test under different scenarios

---

#### **4.4: `scenario_analysis()`**

**Purpose:** Test specific "what if" scenarios

**Example Scenarios:**

**Scenario 1: Market Crash**
```python
Inputs:
  Current Portfolio: $1,000,000
  Price Shock: -15%
  Vol Spike: +200% (from 25% to 75%)
  Spread Widening: +50 bps

Results:
  Direct P&L Impact: -$150,000
  Spread Cost Increase: +$5,000
  New Portfolio Value: $845,000
  
Action: Reduce position size by 20%
```

**Scenario 2: Volatility Spike (Without Price Move)**
```python
Inputs:
  Price Shock: 0%
  Vol Spike: +100% (from 25% to 50%)
  Spread Widening: +20 bps

Results:
  Direct P&L: $0
  Financing Cost Increase: +$20,000 annually
  Decision: Switch 30% of synthetics to cash
```

**Scenario 3: 2008-Style Crisis**
```python
Inputs:
  Price Shock: -40%
  Vol Spike: +300%
  Spread Widening: +200 bps
  Liquidity Dries Up: Bid-ask spreads widen 10x

Results:
  Portfolio Value: -$420,000
  Unable to exit positions
  VaR Exceeded by 4x
  
Action: This is your "worst case" - ensure sizing prevents ruin
```

**Why You Need It:** Understand behavior in extreme markets.

---

#### **4.5: `sharpe_ratio()`**

**Purpose:** Risk-adjusted performance metric

**Formula:**
```
Sharpe Ratio = (Return - Risk Free Rate) / Volatility
```

**Example:**

```python
Strategy Performance:
  Annual Return: 12.3%
  Risk-Free Rate: 4.5%
  Annual Volatility: 8.7%

Sharpe = (0.123 - 0.045) / 0.087
       = 0.078 / 0.087
       = 0.90

Interpretation:
- Sharpe > 1.0 = Good
- Sharpe > 1.5 = Very Good
- Sharpe > 2.0 = Excellent

Your 0.90 is decent, room for improvement
```

**Comparison:**
```
Your Strategy: Sharpe = 0.90
S&P 500: Sharpe = 0.65
Outperformance: 38% better risk-adjusted returns
```

**Why You Need It:** Standard metric for comparing strategies.

---

#### **4.6: `sortino_ratio()`**

**Purpose:** Like Sharpe, but only penalizes DOWNSIDE volatility

**Formula:**
```
Sortino = (Return - Risk Free) / Downside Deviation
```

**Example:**

```python
Your returns:
[2%, -1%, 3%, -2%, 4%, 1%, -0.5%, 2.5%, ...]

Only negative returns:
[-1%, -2%, -0.5%]

Downside Deviation = Std Dev of negative returns = 5.2%

Sortino = (12.3% - 4.5%) / 5.2%
        = 1.5

vs Sharpe = 0.90

Interpretation:
Higher Sortino vs Sharpe means:
- Your losses are smaller/less frequent
- Upside volatility is high (good!)
- Better risk profile than Sharpe suggests
```

**Why You Need It:** More appropriate for asymmetric strategies.

---

#### **4.7: `max_drawdown()`**

**Purpose:** Worst peak-to-trough decline

**Example:**

```python
Portfolio Value Over Time:
$1,000,000 → $1,050,000 → $1,100,000 (peak)
$1,100,000 → $1,080,000 → $1,020,000 (trough)
$1,020,000 → $1,060,000 → $1,150,000 (new peak)

Maximum Drawdown = (Peak - Trough) / Peak
                 = ($1,100,000 - $1,020,000) / $1,100,000
                 = $80,000 / $1,100,000
                 = 7.3%

Interpretation:
"At worst, I was down 7.3% from my previous high"
```

**Why It Matters:**
```
Drawdowns test psychological tolerance:
- 5%: Barely noticeable
- 10%: Uncomfortable
- 20%: Very stressful
- 30%+: Many people quit

Your 7.3% is manageable
```

---

### **What This Component Produces**

**Risk Report:**
```python
{
    'var_95': -5100,
    'cvar_95': -6331,
    'sharpe_ratio': 0.90,
    'sortino_ratio': 1.50,
    'max_drawdown': -0.073,
    'monte_carlo': {
        'mean_final': 1083000,
        'worst_5pct': 781000,
        'prob_loss': 0.234
    },
    'scenarios': {
        'market_crash': -150000,
        'vol_spike': -20000
    }
}
```

---

## **COMPONENT 5: Backtester** (`backtester.py`)

### **What It Does**
Tests if your strategy actually works using historical data.

### **Key Functions Explained**

#### **5.1: `generate_signals()`**

**Purpose:** Convert z-scores into trading signals

**Logic:**
```python
if zscore > 2.0:
    signal = -1  # Short basis (avoid synthetics, use cash)
elif zscore < -2.0:
    signal = 1   # Long basis (use synthetics)
elif abs(zscore) < 0.5:
    signal = 0   # Exit
else:
    signal = previous_signal  # Hold
```

**Example:**

```python
Date        Z-Score  Signal  Action
2024-01-15   -0.3    0       No position
2024-01-16   -1.5    0       Watching...
2024-01-17   -2.2    1       ENTER: Use synthetic (spread is cheap)
2024-01-18   -2.0    1       HOLD
2024-01-19   -1.3    1       HOLD
2024-01-20   -0.4    0       EXIT: Take profit (mean reversion)
2024-01-21    0.8    0       No position
2024-01-22    2.3   -1       ENTER: Use cash (spread is expensive)
2024-01-23    1.9   -1       HOLD
2024-01-24    0.3    0       EXIT: Take profit
```

**Why You Need It:** Systematic rules remove emotion from decisions.

---

#### **5.2: `backtest_strategy()`**

**Purpose:** Calculate actual P&L from following signals

**The Process:**

**Step 1: Set Up Positions**
```python
Date        Price   Signal  Position  
2024-01-17  $150    1       Enter synthetic (1,000 shares)
2024-01-18  $152    1       Hold
2024-01-19  $151    1       Hold
2024-01-20  $153    0       Exit (1,000 shares)
```

**Step 2: Calculate Returns**
```python
Day 1: (152 - 150) / 150 = +1.33%
Day 2: (151 - 152) / 152 = -0.66%
Day 3: (153 - 151) / 151 = +1.32%

Position P&L:
Day 1: 1,000 × $150 × 1.33% = $2,000
Day 2: 1,000 × $152 × -0.66% = -$1,003
Day 3: 1,000 × $151 × 1.32% = $1,993

Total: $2,990
```

**Step 3: Subtract Costs**
```python
Synthetic financing: 50 bps annually = 0.137 bps daily
Position size: $150,000
Cost per day: $150,000 × 0.000137 = $20.55

Total days held: 3
Total cost: 3 × $20.55 = $61.65

Net P&L: $2,990 - $61.65 = $2,928.35
```

**Step 4: Repeat for All Trades**

```python
Trade Summary:
  Number of trades: 23
  Winners: 16 (69.6%)
  Losers: 7 (30.4%)
  
  Average Win: $1,834
  Average Loss: -$892
  
  Total Gross P&L: $23,123
  Total Costs: -$2,456
  Total Net P&L: $20,667
  
  ROI: 20.7% over test period
```

**Why You Need It:** Proves strategy works in real market conditions.

---

#### **5.3: `compare_synthetic_vs_cash()`**

**Purpose:** Show cost savings from optimal selection

**Example:**

```python
Backtest: 100 trades over 2 years

Strategy A: Always Use Synthetics
  Total cost: $125,000
  
Strategy B: Always Use Cash
  Total cost: $118,000
  
Strategy C: Your Adaptive Strategy
  Total cost: $106,000
  
Savings vs Always-Synthetic: $19,000 (15%)
Savings vs Always-Cash: $12,000 (10%)

Conclusion: Adaptive strategy saves $12k-$19k annually
```

**Why You Need It:** Quantifies the value of your system.

---

### **What This Component Produces**

**Backtest Results:**
```python
{
    'total_return': 0.207,  # 20.7%
    'sharpe_ratio': 1.52,
    'max_drawdown': -0.061,
    'num_trades': 23,
    'win_rate': 0.696,
    'avg_win': 1834,
    'avg_loss': -892,
    'cost_savings': 12000,
    'daily_pnl': [120, -45, 230, ...],  # Series
    'cumulative_returns': [1.0, 1.001, 0.999, ...]  # Series
}
```

---

## **COMPONENT 6: Visualizer** (`visualizer.py`)

### **What It Does**
Creates professional charts for presentations/reports.

### **Key Visualizations**

#### **6.1: Z-Score Signal Chart**

Shows when you enter/exit trades:

```
      Z-Score Over Time
  3 |                    *
  2 |               *         ← SHORT signal
  1 |          *                 *
  0 |-----*-----------*------*-------  ← Mean
 -1 |  *                          *
 -2 |*                               ← LONG signal
 -3 |
    |_________________________________
       Time →
    
    ▲ = Long Entry
    ▼ = Short Entry
    ● = Exit
```

**Why You Need It:** Visual proof of timing strategy.

---

#### **6.2: Cumulative Returns Chart**

Compares your strategy vs. benchmarks:

```
    Cumulative Returns
1.3 |                        Your Strategy
1.2 |                   *****
1.1 |            *******
1.0 |     *******              S&P 500
0.9 |*****        *******
    |_________________________________
       Time →
```

**Why You Need It:** Shows outperformance visually.

---

#### **6.3: Monte Carlo Paths**

Shows range of possible outcomes:

```
    Monte Carlo Simulation (10,000 paths)
1.5M |                    *
1.3M |              ********  ← 95th percentile
1.1M |        **************  ← Median
0.9M |    ******************
0.7M |  **              **    ← 5th percentile
     |_________________________________
        Time →
```

**Why You Need It:** Visualizes risk/reward distribution.

---

#### **6.4: Regression Diagnostics**

Shows model fit:

```
Actual vs Predicted Spreads

  |                       
50|              * * *
  |           * * * *
  |        * * * *
  |     * * * * 
  |  * * * *
  |* * *
  |___________________________
     Predicted →
     
R² = 0.67 (good fit)
```

**Why You Need It:** Validates statistical model quality.

---

### **What This Component Produces**

High-quality PNG files:
- `zscore_signals.png`
- `cumulative_returns.png`
- `monte_carlo_paths.png`
- `regression_fit.png`
- `risk_distribution.png`

---

## **HOW EVERYTHING CONNECTS**

```
DATA PIPELINE
    ↓
 [Prices, Returns, Volatility]
    ↓
SYNTHETIC PRICER ← Uses volatility
    ↓
 [Spreads, Costs, Basis]
    ↓
STATISTICAL MODELS ← Finds patterns
    ↓
 [Z-Scores, Forecasts, Predictions]
    ↓
BACKTESTER ← Tests strategy
    ↓
 [Signals, Trades, P&L]
    ↓
RISK ANALYTICS ← Measures risk
    ↓
 [VaR, Sharpe, Max DD]
    ↓
VISUALIZER ← Creates reports
    ↓
 [Charts, Tables, Presentation]
```

---

## **SUMMARY: What Each Component Does**

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Data Pipeline** | Ticker symbols | Prices, returns, volatility | Get raw data |
| **Synthetic Pricer** | Volatility, prices | Financing costs, basis | Calculate economics |
| **Statistical Models** | Basis history | Z-scores, predictions | Find opportunities |
| **Risk Analytics** | Strategy returns | VaR, Sharpe, scenarios | Measure risk |
| **Backtester** | Signals, prices | P&L, trades, performance | Prove it works |
| **Visualizer** | All outputs | Charts, reports | Present findings |

---

Does this breakdown clarify each component? Want me to dive even deeper into any specific calculation or concept?
