# Data settings for fetch_market_data()
DATA_CONFIG = {
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],  # Just 5 stocks for testing
    'start_date': '2023-01-01',
    'end_date': '2025-12-31',
    'benchmark': 'SPY',
}

# Path settings
PATH_CONFIG = {
    'raw_data_path': 'data/raw/',
    'processed_data_path': 'data/processed/',
}

# Synthetic financing parameters
FINANCING_CONFIG = {
    'sofr_rate': 0.045,  # 4.5% SOFR
    'base_spread': 0.0015,  # 15 bps base spread
    'vol_coefficient': 0.002,  # 20 bps per vol point
    'financing_days_per_year': 360,  # Actual/360 convention
}

# Trading parameters
TRADING_CONFIG = {
    'notional_per_trade': 100000,  # $100k per trade
    'transaction_cost': 0.001,  # 0.1% per trade
}

STRESS_TEST_SCENARIOS = {
    'Base Case': {'price_shock': 0.0, 'vol_shock': 1.0, 'spread_shock': 0},
    'Mild Correction': {'price_shock': -0.05, 'vol_shock': 1.2, 'spread_shock': 10},
    'Market Correction': {'price_shock': -0.10, 'vol_shock': 1.5, 'spread_shock': 25},
    'Market Crash': {'price_shock': -0.15, 'vol_shock': 2.0, 'spread_shock': 50},
    'Severe Crisis': {'price_shock': -0.25, 'vol_shock': 2.5, 'spread_shock': 100},
    '2008 Crisis': {'price_shock': -0.40, 'vol_shock': 3.0, 'spread_shock': 200},
    'Vol Spike Only': {'price_shock': 0.0, 'vol_shock': 2.0, 'spread_shock': 30},
    'Liquidity Crisis': {'price_shock': -0.08, 'vol_shock': 2.2, 'spread_shock': 150}
}