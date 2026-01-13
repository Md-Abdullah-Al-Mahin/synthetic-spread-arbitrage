import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_pipeline import DataPipeline


def test_returns_manual():
    """Test returns calculation with manual data"""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    manual_prices = pd.DataFrame({
        'Stock_A': [100.0, 102.0, 101.0, 103.0, 104.0],
    }, index=dates)

    pipeline = DataPipeline()
    returns = pipeline.calculate_returns(manual_prices)

    print("Manual test:")
    print(f"Prices:\n{manual_prices}")
    print(f"Returns:\n{returns}")

    # Verify one calculation
    expected = (102.0 - 100.0) / 100.0
    actual = returns.loc[dates[1], 'Stock_A']
    print(f"Verification: Expected {expected:.4f}, Got {actual:.4f}")

    return returns


def test_returns_real():
    """Test with real data"""
    pipeline = DataPipeline()

    prices = pipeline.fetch_market_data(
        tickers=['AAPL'],
        start_date='2024-06-01',
        end_date='2024-06-05'
    )

    returns = pipeline.calculate_returns(prices)

    print(f"\nReal data test:")
    print(f"Prices: {len(prices)} days")
    print(f"Returns: {len(returns)} days")
    print(f"First returns:\n{returns.head()}")

    return returns