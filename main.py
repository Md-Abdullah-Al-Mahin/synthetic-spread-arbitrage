from src.data_pipeline import DataPipeline
from config import DEFAULT_TICKERS, DEFAULT_START_DATE, DEFAULT_END_DATE

def main():
    """Main execution script"""
    
    print("=== Synthetic Optimizer: Data Pipeline ===\n")
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Fetch market data
    prices = pipeline.fetch_market_data(
        tickers=DEFAULT_TICKERS,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        save_to_disk=True
    )
    
    print("\n=== Data Summary ===")
    print(f"Total trading days: {len(prices)}")
    print(f"Number of stocks: {len(prices.columns)}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices

if __name__ == "__main__":
    prices = main()