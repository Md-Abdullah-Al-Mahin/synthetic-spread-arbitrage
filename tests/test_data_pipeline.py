import sys

sys.path.append('src')

from data_pipeline import DataPipeline


def test_fetch_market_data():
    """Test fetching a small sample of data"""

    # Initialize pipeline
    pipeline = DataPipeline()

    # Test with a few tickers and recent dates
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-01-01'
    end_date = '2024-12-31'

    # Fetch data
    prices = pipeline.fetch_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )

    # Verify results
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

    # Check for missing values
    missing = prices.isnull().sum()
    print(f"Missing values per ticker:\n{missing}")

    return prices