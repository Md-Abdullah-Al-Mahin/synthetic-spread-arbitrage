import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_pipeline import DataPipeline


def test_pipeline():
    """Test the complete pipeline"""
    pipeline = DataPipeline()

    # Step 1: Get prices
    print("Fetching market data...")
    prices = pipeline.fetch_market_data(
        tickers=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-03-01'
    )
    print(f"Prices: {prices.shape}")

    # Step 2: Calculate returns
    print("\nCalculating returns...")
    returns = pipeline.calculate_returns(prices)
    print(f"Returns: {returns.shape}")

    # Step 3: Calculate volatility
    print("\nCalculating volatility...")
    volatility = pipeline.calculate_realized_volatility(returns, window=30)
    print(f"Volatility: {volatility.shape}")

    # Show results
    print("\nResults:")
    for ticker in prices.columns:
        if ticker in volatility.columns:
            current_vol = volatility[ticker].iloc[-1] * 100
            print(f"{ticker}: {current_vol:.1f}% volatility")

    return pipeline