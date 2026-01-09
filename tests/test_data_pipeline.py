import sys
sys.path.append('src')

from data_pipeline import DataPipeline

def test_fetch_market_data():
    """Test fetching a small sample of data"""
    
    # Initialize pipeline
    pipeline = DataPipeline(data_dir="data/raw")
    
    # Test with a few tickers and recent dates
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    # Fetch data
    prices = pipeline.fetch_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        save_to_disk=True
    )
    
    # Verify results
    print("\n--- Verification ---")
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"\nFirst 5 rows:")
    print(prices.head())
    print(f"\nLast 5 rows:")
    print(prices.tail())
    print(f"\nBasic statistics:")
    print(prices.describe())
    
    # Check for missing values
    missing = prices.isnull().sum()
    print(f"\nMissing values per ticker:")
    print(missing)
    
    return prices

if __name__ == "__main__":
    prices = test_fetch_market_data()