import pandas as pd
import numpy as np
import yfinance as yf
import os
from typing import List, Optional, Dict, Any
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
        self.vix = None
        self.liquidity = None

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

    def fetch_vix_data(self,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       force_download: bool = False) -> pd.Series:
        """
        Download VIX (Volatility Index) data
        """
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"data/raw/vix_{start_date}_{end_date}.csv"

        if not force_download and os.path.exists(cache_file):
            print("Loading VIX data from cache")
            self.vix = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze()
            return self.vix

        print("Downloading VIX data...")

        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if not vix_hist.empty:
                self.vix = vix_hist['Close']
                self.vix.index = self.vix.index.tz_localize(None)  # Remove timezone

                # Save to cache
                self.vix.to_csv(cache_file, header=True)
                print(f"VIX data saved: {len(self.vix)} days")
            else:
                print("Warning: No VIX data downloaded")
                self.vix = pd.Series(dtype=float)

        except Exception as e:
            print(f"Error downloading VIX data: {e}")
            self.vix = pd.Series(dtype=float)

        return self.vix

    def fetch_liquidity_data(self,
                             tickers: Optional[List[str]] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             force_download: bool = False) -> pd.DataFrame:
        """
        Estimate liquidity using bid-ask spread approximation
        For simplicity, we'll use volume/price ratio as a proxy
        """
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"data/raw/liquidity_{start_date}_{end_date}.csv"

        if not force_download and os.path.exists(cache_file):
            print("Loading liquidity data from cache")
            self.liquidity = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.liquidity

        print(f"Fetching liquidity data for {len(tickers)} tickers")

        liquidity_data = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)

                if not hist.empty and 'Volume' in hist.columns and 'Close' in hist.columns:
                    # Use volume/dollar volume ratio as liquidity proxy
                    # Higher ratio = more liquid
                    dollar_volume = hist['Volume'] * hist['Close']
                    avg_dollar_volume = dollar_volume.rolling(20).mean()

                    # Normalize and invert (so higher = less liquid, like bid-ask spread)
                    # We use log for better distribution
                    liquidity_metric = 1 / (np.log1p(avg_dollar_volume) + 1)
                    liquidity_data[ticker] = liquidity_metric

                    print(f"  {ticker}: {len(hist)} days")
                else:
                    print(f"  {ticker}: Insufficient data")

            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")
                continue

        if liquidity_data:
            self.liquidity = pd.DataFrame(liquidity_data).dropna()
            self.liquidity.index = self.liquidity.index.tz_localize(None)

            # Save to cache
            self.liquidity.to_csv(cache_file)
            print(f"Liquidity data saved: {self.liquidity.shape}")
        else:
            print("No liquidity data found")
            self.liquidity = pd.DataFrame()

        return self.liquidity

    def calculate_rolling_bid_ask_spread(self,
                                         ticker: str,
                                         window: int = 20) -> pd.Series:
        """
        Calculate rolling bid-ask spread percentage
        This is a more direct measure of liquidity

        Note: Requires high-frequency data, so we'll use a simplified version
        """
        try:
            stock = yf.Ticker(ticker)

            # Get recent market data
            hist = stock.history(period='1mo', interval='1d')

            if 'High' in hist.columns and 'Low' in hist.columns:
                # Simplified bid-ask spread: (High - Low) / ((High + Low) / 2)
                spread_pct = (hist['High'] - hist['Low']) / ((hist['High'] + hist['Low']) / 2)
                rolling_spread = spread_pct.rolling(window).mean()
                return rolling_spread.dropna()

        except Exception as e:
            print(f"Error calculating bid-ask spread for {ticker}: {e}")

        return pd.Series(dtype=float)

    def get_market_data_complete(self,
                                 tickers: Optional[List[str]] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 include_vix: bool = True,
                                 include_liquidity: bool = True,
                                 force_download: bool = False) -> Dict[str, Any]:
        """
        Get complete market data including VIX and liquidity
        """
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        print(f"Fetching complete market data for {len(tickers)} tickers")
        print(f"Date range: {start_date} to {end_date}")

        # Fetch price data
        prices = self.fetch_market_data(tickers, start_date, end_date)

        # Calculate returns and volatility
        returns = self.calculate_returns(prices)
        volatility = self.calculate_realized_volatility(returns, window=30)

        # Fetch dividends
        dividends = self.get_dividend_data(tickers, start_date, end_date)

        # Fetch VIX if requested
        vix_data = None
        if include_vix:
            vix_data = self.fetch_vix_data(start_date, end_date, force_download)

        # Fetch liquidity if requested
        liquidity_data = None
        if include_liquidity:
            liquidity_data = self.fetch_liquidity_data(tickers, start_date, end_date, force_download)

        # Align all data to common dates
        common_dates = volatility.index
        if vix_data is not None:
            vix_data = vix_data.reindex(common_dates).fillna(method='ffill')
        if liquidity_data is not None:
            liquidity_data = liquidity_data.reindex(common_dates).fillna(method='ffill')

        result = {
            'prices': prices,
            'returns': returns,
            'volatility': volatility,
            'dividends': dividends,
            'vix': vix_data,
            'liquidity': liquidity_data,
            'common_dates': common_dates
        }

        print(f"\nData Collection Complete:")
        print(f"  Prices: {prices.shape}")
        print(f"  Volatility: {volatility.shape}")
        print(f"  VIX: {len(vix_data) if vix_data is not None else 0} days")
        print(f"  Liquidity: {liquidity_data.shape if liquidity_data is not None else 'None'}")

        return result