import pandas as pd
import numpy as np
import yfinance as yf
import os
from typing import List, Optional
import config


class DataPipeline:
    def __init__(self):
        """Initialize with basic configuration"""
        self.tickers = config.DATA_CONFIG['tickers']
        self.start_date = config.DATA_CONFIG['start_date']
        self.end_date = config.DATA_CONFIG['end_date']
        self.benchmark = config.DATA_CONFIG['benchmark']

        # Create data directories
        os.makedirs(config.PATH_CONFIG["raw_data_path"], exist_ok=True)
        os.makedirs(config.PATH_CONFIG["processed_data_path"], exist_ok=True)

        # Store downloaded data
        self.prices = None
        self.returns = None
        self.volatility = None
        self.dividends = None

    def fetch_market_data(self,
                          tickers: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Download historical price data for given tickers"""
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"data/raw/market_data_{start_date}_{end_date}.csv"

        print(f"Downloading market data for {len(tickers)} tickers...")

        price_data = {}

        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)

                if not hist.empty:
                    price_data[ticker] = hist['Close']
                    print(f"  {ticker}: {len(hist)} days")
                else:
                    print(f"  No data for {ticker}")

            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")

        if not price_data:
            raise ValueError("No data could be downloaded for any ticker")

        self.prices = pd.DataFrame(price_data)
        self.prices.to_csv(cache_file)
        print(f"Data saved to: {cache_file}")

        return self.prices

    def calculate_returns(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate daily percentage returns"""
        if prices is None and self.prices is None:
            raise ValueError("No price data available. Run fetch_market_data() first.")

        prices = prices if not prices.empty else self.prices

        print(f"Calculating returns for {len(prices.columns)} tickers")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

        # Calculate percentage returns
        self.returns = prices.pct_change().dropna()

        # Clean data
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Save to cache
        returns_file = f"data/processed/daily_returns_{prices.index[0].date()}_{prices.index[-1].date()}.csv"
        self.returns.to_csv(returns_file)
        print(f"Returns data saved to: {returns_file}")

        return self.returns

    def calculate_realized_volatility(self,
                                      returns: Optional[pd.DataFrame] = None,
                                      window: int = 30) -> pd.DataFrame:
        """Calculate rolling volatility (annualized)"""
        if returns is None and self.returns is None:
            raise ValueError("No returns data available. Run calculate_returns() first.")

        returns = returns if not returns.empty else self.returns

        print(f"Calculating {window}-day rolling volatility")

        # Calculate rolling standard deviation and annualize
        daily_vol = returns.rolling(window=window).std()
        self.volatility = daily_vol * np.sqrt(252)
        self.volatility = self.volatility.dropna()

        print(f"Volatility calculated: {self.volatility.index[0].date()} to {self.volatility.index[-1].date()}")

        # Save to cache
        vol_file = f"data/processed/volatility_{window}day.csv"
        self.volatility.to_csv(vol_file)
        print(f"Volatility data saved to: {vol_file}")

        return self.volatility

    def get_dividend_data(self,
                          tickers: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Get dividend payment history"""
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"data/processed/dividends_{start_date}_{end_date}.csv"

        if os.path.exists(cache_file):
            print("Loading dividend data from cache")
            self.dividends = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.dividends

        print(f"Fetching dividend data for {len(tickers)} tickers")
        dividend_data = []

        for ticker in tickers:
            try:
                dividends = yf.Ticker(ticker).dividends

                if dividends.empty:
                    print(f"  {ticker}: No dividend history")
                    continue

                # Filter by date range
                mask = (dividends.index >= pd.Timestamp(start_date)) & \
                       (dividends.index <= pd.Timestamp(end_date))
                dividends = dividends[mask]

                if not dividends.empty:
                    df = pd.DataFrame({
                        'Dividend': dividends.values,
                        'Ticker': ticker
                    }, index=dividends.index.tz_localize(None))
                    df.index.name = 'Date'

                    dividend_data.append(df)
                    print(f"  {ticker}: {len(dividends)} payments")

            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")

        if dividend_data:
            combined = pd.concat(dividend_data)
            self.dividends = pd.pivot_table(
                combined,
                index='Date',
                columns='Ticker',
                values='Dividend',
                aggfunc='sum'
            ).sort_index()

            self.dividends.to_csv(cache_file)
            print(f"Dividend data saved")
        else:
            print("No dividend data found")
            self.dividends = pd.DataFrame()

        return self.dividends

    def calculate_dividend_yield(self,
                                 prices: Optional[pd.DataFrame] = None,
                                 dividends: Optional[pd.DataFrame] = None,
                                 period: str = 'annual') -> pd.DataFrame:
        """Calculate dividend yield for specified period"""
        if prices is None:
            prices = self.prices
        if dividends is None:
            dividends = self.dividends

        if prices is None or dividends is None:
            raise ValueError("Price and dividend data required")

        print(f"Calculating {period} dividend yield")

        # Map period to frequency and multiplier
        freq_map = {
            'annual': ('Y', 1),
            'quarterly': ('Q', 4),
            'monthly': ('M', 12)
        }

        if period not in freq_map:
            raise ValueError("Period must be 'annual', 'quarterly', or 'monthly'")

        freq, multiplier = freq_map[period]

        # Resample data
        resampled_dividends = dividends.resample(freq).sum()
        period_prices = prices.resample(freq).last()

        # Align dates
        common_dates = resampled_dividends.index.intersection(period_prices.index)

        if len(common_dates) == 0:
            print("No overlapping dates for dividend yield calculation")
            return pd.DataFrame()

        resampled_dividends = resampled_dividends.loc[common_dates]
        period_prices = period_prices.loc[common_dates]

        # Calculate yield
        dividend_yield = (resampled_dividends / period_prices) * 100 * multiplier

        print(f"Calculated yield for {len(dividend_yield)} periods")
        return dividend_yield