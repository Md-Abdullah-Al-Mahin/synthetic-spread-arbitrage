import sys

sys.path.append("..")
from data_pipeline import DataPipeline

def test_fetch_market_data():
    # Create pipeline
    pipeline = DataPipeline()

    # Test with default parameters
    print("\nTesting with default parameters...")
    prices = pipeline.fetch_market_data()

    # Test with custom parameters
    print("\nTesting with custom parameters...")
    custom_tickers = ['AAPL', 'MSFT']
    custom_start = '2024-06-01'
    custom_end = '2024-06-30'

    custom_prices = pipeline.fetch_market_data(
        tickers=custom_tickers,
        start_date=custom_start,
        end_date=custom_end
    )

    return prices