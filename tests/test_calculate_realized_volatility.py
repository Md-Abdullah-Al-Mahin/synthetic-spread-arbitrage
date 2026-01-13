import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_pipeline import DataPipeline


def test_volatility_manual():
    """Test with manual data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')

    manual_returns = pd.DataFrame({
        'Low_Vol': np.random.normal(0.0005, 0.01, 252),
        'High_Vol': np.random.normal(0.001, 0.03, 252),
    }, index=dates)

    pipeline = DataPipeline()
    volatility = pipeline.calculate_realized_volatility(manual_returns, window=20)

    print(f"Manual test complete. Shape: {volatility.shape}")
    return volatility


def test_edge_cases():
    """Test edge cases"""
    pipeline = DataPipeline()

    # Test small data
    small_returns = pd.DataFrame({'A': [0.01, -0.02]},
                                 index=pd.date_range('2024-01-01', periods=2))

    # Test constant returns
    const_returns = pd.DataFrame({'B': [0.01] * 10},
                                 index=pd.date_range('2024-01-01', periods=10))

    results = []
    for returns, name in [(small_returns, 'small'), (const_returns, 'constant')]:
        try:
            vol = pipeline.calculate_realized_volatility(returns, window=5)
            results.append(f"{name}: {vol.shape}")
        except Exception as e:
            results.append(f"{name}: error")

    print("Edge cases:", ", ".join(results))
    return True


def test_with_real_data():
    """Test with real data"""
    pipeline = DataPipeline()

    prices = pipeline.fetch_market_data(
        tickers=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-03-01'
    )

    returns = pipeline.calculate_returns(prices)
    volatility = pipeline.calculate_realized_volatility(returns, window=30)

    print(f"Real data: {volatility.shape}")
    print(f"Latest vols: AAPL={volatility.iloc[-1, 0] * 100:.1f}%, MSFT={volatility.iloc[-1, 1] * 100:.1f}%")

    return volatility