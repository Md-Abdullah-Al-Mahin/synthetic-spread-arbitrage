import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Any
import sys
import os
import config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SyntheticPricer:
    """Calculate synthetic financing costs and compare with cash"""

    def __init__(self):
        """Initialize with financing parameters"""
        cfg = config.FINANCING_CONFIG
        self.sofr_rate = cfg.get('sofr_rate', 0.045)
        self.base_spread = cfg.get('base_spread', 0.0015)
        self.vol_coefficient = cfg.get('vol_coefficient', 0.002)
        self.days_per_year = cfg.get('financing_days_per_year', 360)

    def calculate_financing_cost(self,
                                 notional: float = 100000,
                                 days: int = 90,
                                 spread: Optional[float] = None,
                                 sofr_rate: Optional[float] = None) -> Dict[str, float]:
        """Calculate financing cost for synthetic exposure"""
        sofr_rate = sofr_rate or self.sofr_rate
        spread = spread or self.base_spread
        total_rate = sofr_rate + spread

        time_factor = days / self.days_per_year
        total_cost = total_rate * notional * time_factor

        return {
            'total_cost': total_cost,
            'daily_cost': total_cost / days,
            'annual_rate': total_rate,
            'sofr_rate': sofr_rate,
            'spread': spread
        }

    def calculate_multiple_positions(self,
                                     notionals: Dict[str, float],
                                     days: int = 90) -> pd.DataFrame:
        """Calculate financing costs for multiple positions"""
        results = []

        for ticker, notional in notionals.items():
            cost_data = self.calculate_financing_cost(notional=notional, days=days)
            cost_data.update({'ticker': ticker, 'notional': notional})
            results.append(cost_data)

        df = pd.DataFrame(results).set_index('ticker')
        return df

    def estimate_spread_from_volatility(self,
                                        volatility: Union[float, pd.Series, pd.DataFrame],
                                        base_spread: Optional[float] = None,
                                        vol_coefficient: Optional[float] = None) -> Union[float, pd.Series, pd.DataFrame]:
        """Estimate financing spread based on volatility"""
        base_spread = base_spread or self.base_spread
        vol_coefficient = vol_coefficient or self.vol_coefficient

        if isinstance(volatility, (int, float)):
            return base_spread + (vol_coefficient * volatility)
        elif isinstance(volatility, (pd.Series, pd.DataFrame)):
            return base_spread + (vol_coefficient * volatility)
        else:
            raise TypeError(f"Unsupported type: {type(volatility)}")

    def calculate_spread_statistics(self,
                                    volatility: pd.DataFrame,
                                    base_spread: Optional[float] = None,
                                    vol_coefficient: Optional[float] = None) -> pd.DataFrame:
        """Calculate detailed spread statistics for multiple tickers"""
        spreads = self.estimate_spread_from_volatility(volatility, base_spread, vol_coefficient)
        stats_data = []

        for ticker in config.DATA_CONFIG['tickers']:
            ticker_spreads = spreads[ticker].dropna()
            ticker_vols = volatility[ticker].dropna()

            if len(ticker_spreads) > 0:
                stats_data.append({
                    'ticker': ticker,
                    'current_vol': ticker_vols.iloc[-1],
                    'current_spread': ticker_spreads.iloc[-1],
                    'mean_spread': ticker_spreads.mean(),
                    'min_spread': ticker_spreads.min(),
                    'max_spread': ticker_spreads.max(),
                    'spread_range': ticker_spreads.max() - ticker_spreads.min()
                })

        return pd.DataFrame(stats_data).set_index('ticker')

    def calculate_dividend_impact(self,
                                  dividend_yield: pd.DataFrame,
                                  tax_rate: float = 0.30,
                                  position_days: int = 90) -> pd.DataFrame:
        """Calculate dividend impact difference between synthetic and cash positions"""
        tickers = config.DATA_CONFIG['tickers']
        if not tickers:
            print("No tickers found")
            return pd.DataFrame()

        results = []
        for ticker in tickers:
            try:
                dividend_yield_series = dividend_yield[ticker].dropna()

                if dividend_yield_series.empty:
                    continue

                dividend_yield = dividend_yield_series.iloc[-1]
                cash_yield = dividend_yield * (1 - tax_rate)
                synthetic_yield = dividend_yield
                advantage = synthetic_yield - cash_yield

                notional = 100000
                annual_advantage = advantage * notional
                position_advantage = annual_advantage * (position_days / 365)

                results.append({
                    'ticker': ticker,
                    'dividend_yield': dividend_yield,
                    'cash_yield': cash_yield,
                    'synthetic_yield': synthetic_yield,
                    'advantage': advantage,
                    'annual_advantage': annual_advantage,
                    'position_advantage': position_advantage
                })

            except Exception as e:
                print(f"Error with {ticker}: {str(e)[:50]}")
                continue

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results).set_index('ticker')
        
        print(f"Total annual advantage: ${result_df['annual_advantage'].sum():,.0f}")

        return result_df

    def calculate_net_financing_cost(self,
                                     notional: float,
                                     days: int,
                                     volatility: float,
                                     dividend_yield: float,
                                     tax_rate: float = 0.30,
                                     sofr_rate: Optional[float] = None,
                                     base_spread: Optional[float] = None,
                                     vol_coefficient: Optional[float] = None) -> Dict[str, float]:
        """Calculate net financing cost including dividend impact"""
        spread = self.estimate_spread_from_volatility(volatility, base_spread, vol_coefficient)
        financing_data = self.calculate_financing_cost(notional=notional, days=days, spread=spread,sofr_rate=sofr_rate)

        cash_yield = dividend_yield * (1 - tax_rate)
        synthetic_yield = dividend_yield
        advantage = synthetic_yield - cash_yield
        annual_advantage = advantage * notional
        position_advantage = annual_advantage * (days / 365)

        net_cost = financing_data['total_cost'] - position_advantage

        return {
            'gross_financing_cost': financing_data['total_cost'],
            'dividend_advantage': position_advantage,
            'net_financing_cost': net_cost,
            'financing_rate': financing_data['annual_rate'],
            'dividend_yield': dividend_yield,
            'advantage_rate': advantage
        }

    def calculate_total_cost_of_carry(self,
                                      ticker: str,
                                      price: float,
                                      volatility: float,
                                      dividend_yield: Optional[float] = None,
                                      dividends: Optional[pd.DataFrame] = None,
                                      tax_rate: float = 0.30,
                                      days: int = 90,
                                      notional: float = 100000,
                                      sofr_rate: Optional[float] = None,
                                      cash_borrow_rate: Optional[float] = None,
                                      transaction_cost: float = 0.001) -> Dict[str, Any]:
        """Compare all-in costs: synthetic vs cash"""
        sofr_rate = sofr_rate or self.sofr_rate
        cash_borrow_rate = cash_borrow_rate or sofr_rate

        # Calculate dividend yield if not provided
        if dividend_yield is None and dividends is not None and ticker in dividends.columns:
            dividend_series = dividends[ticker].dropna()
            if not dividend_series.empty:
                current_date = dividends.index[-1]
                one_year_ago = current_date - pd.DateOffset(years=1)
                recent_dividends = dividend_series[dividend_series.index >= one_year_ago]
                annual_dividend = recent_dividends.sum()
                dividend_yield = annual_dividend / price if price > 0 else 0
        dividend_yield = dividend_yield or 0.0

        # Synthetic costs
        spread = self.estimate_spread_from_volatility(volatility)
        synthetic_rate = sofr_rate + spread
        financing_data = self.calculate_financing_cost(notional, days, spread, sofr_rate)

        cash_yield = dividend_yield * (1 - tax_rate)
        synthetic_yield = dividend_yield
        dividend_advantage = synthetic_yield - cash_yield
        annual_advantage = dividend_advantage * notional
        period_advantage = annual_advantage * (days / 365)

        transaction_cost_amount = transaction_cost * notional
        total_synthetic_cost = financing_data['total_cost'] - period_advantage + transaction_cost_amount

        # Cash costs
        opportunity_cost = self.calculate_financing_cost(notional, days, 0, sofr_rate)['total_cost']
        borrow_cost = self.calculate_financing_cost(notional, days, 0, cash_borrow_rate)['total_cost']
        total_cash_long = opportunity_cost + transaction_cost_amount
        total_cash_short = borrow_cost + transaction_cost_amount

        # Determine cheapest
        costs = [
            ('SYNTHETIC', total_synthetic_cost),
            ('CASH_LONG', total_cash_long),
            ('CASH_SHORT', total_cash_short)
        ]
        cheapest = min(costs, key=lambda x: x[1])

        return {
            'ticker': ticker,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'price': price,
            'volatility': volatility,
            'dividend_yield': dividend_yield,
            'estimated_spread': spread,
            'synthetic_rate': synthetic_rate,
            'synthetic_cost': total_synthetic_cost,
            'cash_long_cost': total_cash_long,
            'cash_short_cost': total_cash_short,
            'savings_vs_long': total_cash_long - total_synthetic_cost,
            'savings_vs_short': total_cash_short - total_synthetic_cost,
            'recommendation': cheapest[0],
            'basis': synthetic_rate - cash_borrow_rate
        }

    def batch_cost_analysis(self,
                            tickers: List[str],
                            prices: pd.DataFrame,
                            volatilities: pd.DataFrame,
                            dividends: Optional[pd.DataFrame] = None,
                            **kwargs) -> pd.DataFrame:
        """Run cost analysis for multiple tickers"""
        results = []

        for ticker in tickers:
            try:
                if ticker in prices.columns and ticker in volatilities.columns:
                    price = prices[ticker].iloc[-1]
                    volatility = volatilities[ticker].iloc[-1]

                    ticker_dividends = None
                    if dividends is not None and ticker in dividends.columns:
                        ticker_dividends = dividends[[ticker]]

                    result = self.calculate_total_cost_of_carry(
                        ticker=ticker,
                        price=price,
                        volatility=volatility,
                        dividends=ticker_dividends,
                        **kwargs
                    )
                    results.append(result)

            except Exception as e:
                print(f"Error with {ticker}: {str(e)[:50]}")

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        recs = results_df['recommendation'].value_counts()
        print(f"Recommendations: {recs.to_dict()}")

        return results_df

    def calculate_basis(self,
                        synthetic_rate: float,
                        cash_rate: float) -> float:
        """Calculate basis between synthetic and cash rates"""
        return synthetic_rate - cash_rate

    def calculate_historical_basis(self,
                                   tickers: List[str],
                                   prices: pd.DataFrame,
                                   volatilities: pd.DataFrame,
                                   dividends: Optional[pd.DataFrame] = None,
                                   sofr_rates: Optional[pd.Series] = None,
                                   cash_rates: Optional[pd.Series] = None,
                                   days: int = 90,
                                   tax_rate: float = 0.30) -> pd.DataFrame:
        """Calculate historical basis over time"""
        basis_data = []
        for ticker in tickers:
            if ticker not in prices.columns or ticker not in volatilities.columns:
                continue

            ticker_prices = prices[ticker].dropna()
            ticker_vols = volatilities[ticker].dropna()
            common_dates = ticker_prices.index.intersection(ticker_vols.index)

            for date in common_dates:
                try:
                    price = ticker_prices.loc[date]
                    vol = ticker_vols.loc[date]

                    sofr = self.sofr_rate
                    if sofr_rates is not None and date in sofr_rates.index:
                        sofr = sofr_rates.loc[date]

                    cash = sofr
                    if cash_rates is not None and date in cash_rates.index:
                        cash = cash_rates.loc[date]

                    spread = self.estimate_spread_from_volatility(vol)
                    synthetic_rate = sofr + spread

                    # Dividend adjustment
                    div_yield = 0.0
                    if dividends is not None and ticker in dividends.columns:
                        one_year_ago = date - pd.DateOffset(years=1)
                        recent_divs = dividends[ticker][
                            (dividends[ticker].index >= one_year_ago) &
                            (dividends[ticker].index <= date)
                            ]
                        div_yield = recent_divs.sum() / price if price > 0 else 0

                    cash_yield = div_yield * (1 - tax_rate)
                    synth_yield = div_yield
                    div_advantage = synth_yield - cash_yield
                    effective_synthetic_rate = synthetic_rate - div_advantage
                    basis = effective_synthetic_rate - cash

                    basis_data.append({
                        'date': date,
                        'ticker': ticker,
                        'price': price,
                        'volatility': vol,
                        'synthetic_rate': synthetic_rate,
                        'effective_synthetic_rate': effective_synthetic_rate,
                        'cash_rate': cash,
                        'basis': basis,
                        'dividend_yield': div_yield,
                        'spread': spread
                    })
                except:
                    continue

        if not basis_data:
            return pd.DataFrame()

        basis_df = pd.DataFrame(basis_data)
        pivot_basis = basis_df.pivot(index='date', columns='ticker', values='basis')
        print(f"Calculated basis for {len(pivot_basis.columns)} tickers, {len(pivot_basis)} dates")

        return basis_df

    def generate_basis_signals(self,
                               basis_series: pd.Series,
                               entry_threshold: float = 0.01,
                               exit_threshold: float = 0.0025,
                               lookback_days: int = 252) -> pd.DataFrame:
        """Generate trading signals based on basis levels"""
        signals = pd.DataFrame(index=basis_series.index)
        signals['basis'] = basis_series
        signals['mean'] = basis_series.rolling(lookback_days, min_periods=20).mean()
        signals['std'] = basis_series.rolling(lookback_days, min_periods=20).std()
        signals['zscore'] = (basis_series - signals['mean']) / signals['std']

        signals['signal'] = 0
        signals.loc[basis_series < -entry_threshold, 'signal'] = 1  # Long basis (buy synthetic)
        signals.loc[basis_series > entry_threshold, 'signal'] = -1  # Short basis (sell synthetic)
        signals.loc[basis_series.abs() < exit_threshold, 'signal'] = 0
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)

        print(f"Generated signals: {(signals['position'] != 0).sum()} active days")
        return signals

    def run_full_pricing_analysis(self,
                                  data: Dict[str, Any],
                                  days: int = 90,
                                  notional: float = 100000,
                                  tax_rate: float = 0.30,
                                  transaction_cost: float = 0.001,
                                  include_historical_basis: bool = True,
                                  generate_signals: bool = True,
                                  entry_threshold: float = 0.01,
                                  exit_threshold: float = 0.0025) -> Dict[str, Any]:
        """
        Run complete synthetic pricing analysis
        
        Takes output from DataPipeline and produces comprehensive pricing comparison
        
        Parameters:
        -----------
        data : Dict from DataPipeline.run_full_pipeline()
        days : Holding period for cost calculation (default: 90)
        notional : Position size in dollars (default: 100,000)
        tax_rate : Tax rate on dividends (default: 30%)
        transaction_cost : Transaction cost as % of notional (default: 0.1%)
        include_historical_basis : Calculate basis over time (default: True)
        generate_signals : Generate trading signals (default: True)
        entry_threshold : Basis threshold for signal entry (default: 1%)
        exit_threshold : Basis threshold for signal exit (default: 0.25%)
        
        Returns:
        --------
        Dictionary containing:
            - current_analysis: DataFrame with current cost comparison for all tickers
            - spread_stats: DataFrame with spread statistics by ticker
            - dividend_impact: DataFrame with dividend advantage analysis
            - historical_basis: DataFrame with basis over time (if requested)
            - signals: Dict of DataFrames with trading signals per ticker (if requested)
            - summary: Dict with aggregate statistics
        """

        print("\nRunning full pricing analysis...")
        print(f"Parameters: {days} days, ${notional:,.0f} notional, {tax_rate:.0%} tax rate\n")

        # Extract data components
        prices = data['prices']
        returns = data['returns']
        volatilities = data['volatility']
        dividends = data['dividends']

        tickers = list(volatilities.columns)
        print(f"Analyzing {len(tickers)} tickers")

        # Current cost analysis
        print("Calculating current costs...")
        current_analysis = self.batch_cost_analysis(
            tickers=tickers,
            prices=prices,
            volatilities=volatilities,
            dividends=dividends,
            days=days,
            notional=notional,
            tax_rate=tax_rate,
            transaction_cost=transaction_cost
        )

        # Spread statistics
        print("Calculating spread statistics...")
        spread_stats = self.calculate_spread_statistics(volatilities)

        # Dividend impact
        print("Analyzing dividend impact...")
        dividend_impact = pd.DataFrame()
        if not dividends.empty:
            dividend_impact = self.calculate_dividend_impact(
                prices=prices,
                dividends=dividends,
                tax_rate=tax_rate,
                position_days=days
            )

        # Historical basis
        historical_basis = pd.DataFrame()
        if include_historical_basis:
            print("Calculating historical basis...")
            historical_basis = self.calculate_historical_basis(
                tickers=tickers,
                prices=prices,
                volatilities=volatilities,
                dividends=dividends if not dividends.empty else None,
                sofr_rates=None,
                cash_rates=None,
                days=days,
                tax_rate=tax_rate
            )

            # Save to file
            if not historical_basis.empty:
                cache_file = f"{config.PATH_CONFIG['processed_data_path']}/historical_basis.csv"
                historical_basis.to_csv(cache_file, index=False)
                print(f"Saved historical basis to {cache_file}")

        # Generate signals
        signals_dict = {}
        if generate_signals and not historical_basis.empty:
            print("Generating trading signals...")
            for ticker in tickers:
                ticker_basis = historical_basis[historical_basis['ticker'] == ticker].set_index('date')['basis']
                if len(ticker_basis) > 50:  # Minimum data requirement
                    signals = self.generate_basis_signals(
                        basis_series=ticker_basis,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold
                    )
                    signals_dict[ticker] = signals

        # Create summary statistics
        summary = {
            'num_tickers': len(tickers),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'holding_period_days': days,
            'notional': notional,
            'tax_rate': tax_rate,
        }

        if not current_analysis.empty:
            summary.update({
                'avg_synthetic_cost': current_analysis['synthetic_cost'].mean(),
                'avg_cash_long_cost': current_analysis['cash_long_cost'].mean(),
                'avg_basis': current_analysis['basis'].mean(),
                'recommendations': current_analysis['recommendation'].value_counts().to_dict(),
                'total_savings_vs_long': current_analysis['savings_vs_long'].sum(),
            })

        if not spread_stats.empty:
            summary['avg_spread'] = spread_stats['mean_spread'].mean()
            summary['spread_range'] = {
                'min': spread_stats['min_spread'].min(),
                'max': spread_stats['max_spread'].max()
            }

        if not dividend_impact.empty:
            summary['total_dividend_advantage'] = dividend_impact['annual_advantage'].sum()

        # Package results
        result = {
            'current_analysis': current_analysis,
            'spread_stats': spread_stats,
            'dividend_impact': dividend_impact,
            'historical_basis': historical_basis,
            'signals': signals_dict,
            'summary': summary
        }

        # Print summary
        print("PRICING ANALYSIS COMPLETE")
        print(f"Tickers analyzed: {summary['num_tickers']}")
        
        if not current_analysis.empty:
            print(f"\nCurrent Analysis:")
            print(f"  Avg synthetic cost: ${summary['avg_synthetic_cost']:,.0f}")
            print(f"  Avg cash long cost: ${summary['avg_cash_long_cost']:,.0f}")
            print(f"  Avg basis: {summary['avg_basis']:.2%}")
            print(f"  Recommendations: {summary['recommendations']}")
            print(f"  Total potential savings: ${summary['total_savings_vs_long']:,.0f}")

        if not spread_stats.empty:
            print(f"\nSpread Statistics:")
            print(f"  Average spread: {summary['avg_spread']:.2%}")
            print(f"  Spread range: {summary['spread_range']['min']:.2%} - {summary['spread_range']['max']:.2%}")

        if not dividend_impact.empty:
            print(f"\nDividend Impact:")
            print(f"  Total annual advantage: ${summary['total_dividend_advantage']:,.0f}")

        if not historical_basis.empty:
            print(f"\nHistorical Basis:")
            print(f"  Data points: {len(historical_basis)}")
            print(f"  Date range: {historical_basis['date'].min().date()} to {historical_basis['date'].max().date()}")

        if signals_dict:
            print(f"\nTrading Signals:")
            print(f"  Signals generated for {len(signals_dict)} tickers")
            total_active = sum((sig['position'] != 0).sum() for sig in signals_dict.values())
            print(f"  Total active signal days: {total_active}")

        return result