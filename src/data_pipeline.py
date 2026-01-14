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
                          end_date: Optional[str] = None,
                          force_download: bool = False) -> pd.DataFrame:
        """Download historical price data for given tickers"""
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"{config.PATH_CONFIG['raw_data_path']}/market_data_{start_date}_{end_date}.csv"

        # Check cache first
        if not force_download and os.path.exists(cache_file):
            print(f"Loading market data from cache")
            self.prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.prices

        print(f"Downloading market data for {len(tickers)} tickers...")

        price_data = {}

        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)

                if not hist.empty:
                    price_data[ticker] = hist['Close']
                    print(f"  {ticker}: {len(hist)} days")
                else:
                    print(f"  {ticker}: No data")

            except Exception as e:
                print(f"  {ticker}: Error - {e}")

        if not price_data:
            raise ValueError("No data could be downloaded for any ticker")

        self.prices = pd.DataFrame(price_data)
        self.prices.to_csv(cache_file)
        print(f"Saved to {cache_file}")

        return self.prices

    def calculate_returns(self, 
                         prices: Optional[pd.DataFrame] = None,
                         force_calculate: bool = False) -> pd.DataFrame:
        """Calculate daily percentage returns"""
        prices = prices if prices is not None else self.prices
        
        if prices is None:
            raise ValueError("No price data available. Run fetch_market_data() first.")

        # Generate cache filename based on price data date range
        start_date = prices.index[0].date()
        end_date = prices.index[-1].date()
        returns_file = f"{config.PATH_CONFIG['processed_data_path']}/daily_returns_{start_date}_{end_date}.csv"

        # Check cache first
        if not force_calculate and os.path.exists(returns_file):
            print(f"Loading returns from cache")
            self.returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            return self.returns

        print(f"Calculating returns for {len(prices.columns)} tickers")

        # Calculate percentage returns
        self.returns = prices.pct_change().dropna()

        # Clean data
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Save to cache
        self.returns.to_csv(returns_file)
        print(f"Saved to {returns_file}")

        return self.returns

    def calculate_realized_volatility(self,
                                      returns: Optional[pd.DataFrame] = None,
                                      window: int = 30,
                                      force_calculate: bool = False) -> pd.DataFrame:
        """Calculate rolling volatility (annualized)"""
        returns = returns if returns is not None else self.returns
        
        if returns is None:
            raise ValueError("No returns data available. Run calculate_returns() first.")

        # Generate cache filename based on returns data date range
        start_date = returns.index[0].date()
        end_date = returns.index[-1].date()
        vol_file = f"{config.PATH_CONFIG['processed_data_path']}/volatility_{window}day_{start_date}_{end_date}.csv"

        # Check cache first
        if not force_calculate and os.path.exists(vol_file):
            print(f"Loading volatility from cache")
            self.volatility = pd.read_csv(vol_file, index_col=0, parse_dates=True)
            return self.volatility

        print(f"Calculating {window}-day rolling volatility...")

        # Calculate rolling standard deviation and annualize
        daily_vol = returns.rolling(window=window).std()
        self.volatility = daily_vol * np.sqrt(252)
        self.volatility = self.volatility.dropna()

        # Save to cache
        self.volatility.to_csv(vol_file)
        print(f"Saved to {vol_file}")

        return self.volatility

    def get_dividend_data(self,
                          tickers: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          force_download: bool = False) -> pd.DataFrame:
        """Get dividend payment history"""
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"{config.PATH_CONFIG['processed_data_path']}/dividends_{start_date}_{end_date}.csv"

        # Check cache first
        if not force_download and os.path.exists(cache_file):
            print("Loading dividend data from cache")
            self.dividends = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.dividends

        print(f"Fetching dividend data for {len(tickers)} tickers...")
        dividend_data = []

        for ticker in tickers:
            try:
                dividends = yf.Ticker(ticker).dividends

                if dividends.empty:
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
                print(f"  {ticker}: Error - {e}")

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
            print(f"Saved dividend data")
        else:
            print("No dividend data found")
            self.dividends = pd.DataFrame()

        return self.dividends

    def calculate_dividend_yield(self,
                                 prices: Optional[pd.DataFrame] = None,
                                 dividends: Optional[pd.DataFrame] = None,
                                 period: str = 'annual') -> pd.DataFrame:
        """Calculate dividend yield for specified period"""
        prices = prices if prices is not None else self.prices
        dividends = dividends if dividends is not None else self.dividends

        if prices is None or dividends is None:
            raise ValueError("Price and dividend data required")

        print(f"Calculating {period} dividend yield...")

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
            print("No overlapping dates")
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
        """Download VIX (Volatility Index) data"""
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"{config.PATH_CONFIG['raw_data_path']}/vix_{start_date}_{end_date}.csv"

        # Check cache first
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
                self.vix.index = self.vix.index.tz_localize(None)

                self.vix.to_csv(cache_file, header=True)
                print(f"Saved VIX data: {len(self.vix)} days")
            else:
                print("No VIX data downloaded")
                self.vix = pd.Series(dtype=float)

        except Exception as e:
            print(f"Error downloading VIX: {e}")
            self.vix = pd.Series(dtype=float)

        return self.vix

    def fetch_liquidity_data(self,
                             tickers: Optional[List[str]] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             force_download: bool = False) -> pd.DataFrame:
        """Estimate liquidity using volume-based proxy"""
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        cache_file = f"{config.PATH_CONFIG['raw_data_path']}/liquidity_{start_date}_{end_date}.csv"

        # Check cache first
        if not force_download and os.path.exists(cache_file):
            print("Loading liquidity data from cache")
            self.liquidity = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.liquidity

        print(f"Fetching liquidity data for {len(tickers)} tickers...")

        liquidity_data = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)

                if not hist.empty and 'Volume' in hist.columns and 'Close' in hist.columns:
                    dollar_volume = hist['Volume'] * hist['Close']
                    avg_dollar_volume = dollar_volume.rolling(20).mean()
                    liquidity_metric = 1 / (np.log1p(avg_dollar_volume) + 1)
                    liquidity_data[ticker] = liquidity_metric
                    print(f"  {ticker}: {len(hist)} days")

            except Exception as e:
                print(f"  {ticker}: Error - {e}")

        if liquidity_data:
            self.liquidity = pd.DataFrame(liquidity_data).dropna()
            self.liquidity.index = self.liquidity.index.tz_localize(None)

            self.liquidity.to_csv(cache_file)
            print(f"Saved liquidity data")
        else:
            print("No liquidity data found")
            self.liquidity = pd.DataFrame()

        return self.liquidity

    def run_full_pipeline(self,
                          tickers: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          volatility_window: int = 30,
                          include_vix: bool = True,
                          include_liquidity: bool = True,
                          calculate_div_yield: bool = True,
                          force_download: bool = False) -> Dict[str, Any]:
        """
        Run complete data pipeline: fetch and process all market data
        
        Parameters:
        -----------
        tickers : List of stock symbols
        start_date : Start date for historical data
        end_date : End date for historical data
        volatility_window : Rolling window for volatility calculation (default: 30)
        include_vix : Whether to fetch VIX data (default: True)
        include_liquidity : Whether to fetch liquidity metrics (default: True)
        calculate_div_yield : Whether to calculate dividend yields (default: True)
        force_download : Force fresh download, ignore cache (default: False)
        
        Returns:
        --------
        Dictionary containing:
            - prices: DataFrame with daily closing prices
            - returns: DataFrame with daily percentage returns
            - volatility: DataFrame with annualized rolling volatility
            - dividends: DataFrame with dividend payment history
            - dividend_yield: DataFrame with annual dividend yields
            - vix: Series with VIX index values
            - liquidity: DataFrame with liquidity metrics
            - metadata: Dict with summary statistics
        """
        
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        print(f"\nRunning full data pipeline...")
        print(f"Tickers: {tickers}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Force download: {force_download}\n")
        
        # Fetch price data
        prices = self.fetch_market_data(tickers, start_date, end_date, force_download)
        
        # Calculate returns
        returns = self.calculate_returns(prices, force_calculate=force_download)
        
        # Calculate volatility
        volatility = self.calculate_realized_volatility(returns, window=volatility_window, force_calculate=force_download)
        
        # Fetch dividends
        dividends = self.get_dividend_data(tickers, start_date, end_date, force_download)
        
        # Calculate dividend yield (optional)
        dividend_yield = None
        if calculate_div_yield and not dividends.empty:
            dividend_yield = self.calculate_dividend_yield(prices, dividends, period='annual')
        
        # Fetch VIX (optional)
        vix_data = None
        if include_vix:
            vix_data = self.fetch_vix_data(start_date, end_date, force_download)
        
        # Fetch liquidity (optional)
        liquidity_data = None
        if include_liquidity:
            liquidity_data = self.fetch_liquidity_data(tickers, start_date, end_date, force_download)
        
        # Align all data to common dates
        common_dates = volatility.index
        
        if vix_data is not None and not vix_data.empty:
            vix_data = vix_data.reindex(common_dates).ffill()
        
        if liquidity_data is not None and not liquidity_data.empty:
            liquidity_data = liquidity_data.reindex(common_dates).ffill()
        
        # Create metadata summary
        metadata = {
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': len(common_dates),
            'volatility_window': volatility_window,
            'num_tickers': len(tickers),
            'avg_volatility': volatility.mean().to_dict(),
            'avg_returns': returns.mean().to_dict(),
            'total_dividends': dividends.sum().to_dict() if not dividends.empty else {},
        }
        
        # Package results
        result = {
            'prices': prices,
            'returns': returns,
            'volatility': volatility,
            'dividends': dividends,
            'dividend_yield': dividend_yield,
            'vix': vix_data,
            'liquidity': liquidity_data,
            'common_dates': common_dates,
            'metadata': metadata
        }
        
        # Print summary
        print("\nPipeline complete:")
        print(f"  Prices: {prices.shape}")
        print(f"  Returns: {returns.shape}")
        print(f"  Volatility: {volatility.shape}")
        if not dividends.empty:
            print(f"  Dividends: {dividends.shape}")
        if dividend_yield is not None:
            print(f"  Dividend Yield: {dividend_yield.shape}")
        if vix_data is not None:
            print(f"  VIX: {len(vix_data)} days")
        if liquidity_data is not None:
            print(f"  Liquidity: {liquidity_data.shape}")
        print()
        
        return result