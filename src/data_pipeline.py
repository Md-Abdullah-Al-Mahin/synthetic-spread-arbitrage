import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional
import os

class DataPipeline:
    """Handles all data fetching and preprocessing"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data pipeline
        
        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_market_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        save_to_disk: bool = True
    ) -> pd.DataFrame:
        """
        Download historical stock prices from Yahoo Finance
        """
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data = {}
        failed_tickers = []
        
        # Download each ticker separately for better error handling
        for ticker in tickers:
            try:
                print(f"  Downloading {ticker}...", end=" ")
                
                # Download data using yfinance
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                # Check if we got data
                if df.empty:
                    print(f"❌ No data available")
                    failed_tickers.append(ticker)
                    continue
                
                # Use adjusted close price (accounts for splits and dividends)
                all_data[ticker] = df['Close']
                print(f"✓ {len(df)} days")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                failed_tickers.append(ticker)
                continue
        
        # Combine all tickers into one DataFrame
        if not all_data:
            raise ValueError("No data was successfully downloaded for any ticker")
        
        price_df = pd.DataFrame(all_data)
        
        # Clean up the data
        price_df.index.name = 'Date'
        price_df = price_df.sort_index()
        
        # Forward fill missing values (holidays/weekends)
        price_df = price_df.fillna(method='ffill')
        
        print(f"\n✓ Successfully downloaded {len(all_data)} tickers")
        if failed_tickers:
            print(f"⚠ Failed tickers: {failed_tickers}")
        
        # Save to disk if requested
        if save_to_disk:
            filepath = os.path.join(self.data_dir, f"prices_{start_date}_to_{end_date}.csv")
            price_df.to_csv(filepath)
            print(f"💾 Saved to: {filepath}")
        
        return price_df
    
    def load_market_data(self, filename: str) -> pd.DataFrame:
        """
        Load previously saved market data
        
        Args:
            filename: Name of the CSV file in data/raw/
            
        Returns:
            DataFrame with price data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} tickers from {filename}")
        
        return df