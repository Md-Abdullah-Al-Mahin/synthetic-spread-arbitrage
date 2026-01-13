import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Any
import sys
import os
import config

# Add config
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

        print(f"Financing Cost:")
        print(f"  Notional: ${notional:,.2f} | Days: {days}")
        print(f"  Rate: {total_rate:.2%} (SOFR: {sofr_rate:.2%} + Spread: {spread:.2%})")
        print(f"  Total: ${total_cost:,.2f} | Daily: ${total_cost / days:,.2f}")

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
        total_cost = df['total_cost'].sum()
        total_notional = df['notional'].sum()

        print(f"\nSummary for {len(notionals)} positions:")
        print(f"  Total Notional: ${total_notional:,.2f}")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print(f"  Avg Rate: {(total_cost / total_notional) * (self.days_per_year / days):.2%}")

        return df

    def estimate_spread_from_volatility(self,
                                        volatility: Union[float, pd.Series, pd.DataFrame],
                                        base_spread: Optional[float] = None,
                                        vol_coefficient: Optional[float] = None) -> Union[
        float, pd.Series, pd.DataFrame]:
        """Estimate financing spread based on volatility"""
        base_spread = base_spread or self.base_spread
        vol_coefficient = vol_coefficient or self.vol_coefficient

        print(f"Spread Estimation:")
        print(f"  Base: {base_spread:.2%} | Vol Coef: {vol_coefficient:.2%}")

        if isinstance(volatility, (int, float)):
            spread = base_spread + (vol_coefficient * volatility)
            print(f"  Vol: {volatility:.1%} → Spread: {spread:.2%}")
            return spread

        elif isinstance(volatility, pd.Series):
            spread = base_spread + (vol_coefficient * volatility)
            print(f"  Vol Range: {volatility.min():.1%}-{volatility.max():.1%}")
            print(f"  Spread Range: {spread.min():.2%}-{spread.max():.2%}")
            return spread

        elif isinstance(volatility, pd.DataFrame):
            spread = base_spread + (vol_coefficient * volatility)
            print(f"  Shape: {volatility.shape} → {spread.shape[1]} tickers")
            return spread

        raise TypeError(f"Unsupported type: {type(volatility)}")

    def calculate_spread_statistics(self,
                                    volatility: pd.DataFrame,
                                    base_spread: Optional[float] = None,
                                    vol_coefficient: Optional[float] = None) -> pd.DataFrame:
        """Calculate detailed spread statistics for multiple tickers"""
        spreads = self.estimate_spread_from_volatility(volatility, base_spread, vol_coefficient)
        stats_data = []

        for ticker in spreads.columns:
            ticker_spreads = spreads[ticker].dropna()
            ticker_vols = volatility[ticker].dropna()

            if len(ticker_spreads) > 0:
                stats_data.append({
                    'Ticker': ticker,
                    'Current Vol': ticker_vols.iloc[-1] * 100,
                    'Current Spread': ticker_spreads.iloc[-1] * 100,
                    'Mean Spread': ticker_spreads.mean() * 100,
                    'Min Spread': ticker_spreads.min() * 100,
                    'Max Spread': ticker_spreads.max() * 100,
                    'Spread Range': (ticker_spreads.max() - ticker_spreads.min()) * 100,
                    'Spread/Vol Ratio': ticker_spreads.mean() / ticker_vols.mean() if ticker_vols.mean() > 0 else 0
                })

        stats_df = pd.DataFrame(stats_data).set_index('Ticker')
        print("\nSpread Statistics:")
        print(stats_df.round(3))
        return stats_df

    def calculate_dividend_impact(self,
                                  prices: pd.DataFrame,
                                  dividends: pd.DataFrame,
                                  tax_rate: float = 0.30,
                                  position_days: int = 90) -> pd.DataFrame:
        """Calculate dividend impact difference between synthetic and cash positions"""
        print(f"Dividend Impact Analysis (Tax: {tax_rate:.0%})")

        common_tickers = list(set(prices.columns) & set(dividends.columns))
        if not common_tickers:
            print("No common tickers found")
            return pd.DataFrame()

        results = []
        for ticker in common_tickers:
            try:
                price_series = prices[ticker].dropna()
                dividend_series = dividends[ticker].dropna()

                if price_series.empty or dividend_series.empty:
                    continue

                current_date = price_series.index[-1]
                one_year_ago = current_date - pd.DateOffset(years=1)
                recent_dividends = dividend_series[dividend_series.index >= one_year_ago]
                annual_dividend = recent_dividends.sum()
                current_price = price_series.iloc[-1]

                dividend_yield = annual_dividend / current_price
                cash_yield = dividend_yield * (1 - tax_rate)
                synthetic_yield = dividend_yield
                advantage = synthetic_yield - cash_yield

                notional = 100000
                annual_advantage = advantage * notional
                daily_advantage = annual_advantage / 365
                position_advantage = annual_advantage * (position_days / 365)

                results.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'annual_dividend': annual_dividend,
                    'dividend_yield': dividend_yield,
                    'cash_yield': cash_yield,
                    'synthetic_yield': synthetic_yield,
                    'advantage': advantage,
                    'annual_advantage': annual_advantage,
                    'daily_advantage': daily_advantage,
                    'position_advantage': position_advantage
                })

                print(f"{ticker}: Yield: {dividend_yield:.2%} → Adv: {advantage:.2%} (${annual_advantage:.0f}/yr)")

            except Exception as e:
                print(f"Error with {ticker}: {str(e)[:50]}")
                continue

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results).set_index('ticker')
        sorted_df = result_df.sort_values('advantage', ascending=False)

        print("\nTop Advantages:")
        for i, (ticker, row) in enumerate(sorted_df.head().iterrows(), 1):
            print(f"{i}. {ticker}: {row['advantage']:.2%} (${row['annual_advantage']:.0f})")

        total_advantage = result_df['annual_advantage'].sum()
        print(f"\nTotal Annual Advantage: ${total_advantage:.0f}")

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
        print(f"\nNet Financing Cost Analysis")
        print("-" * 40)

        spread = base_spread or self.estimate_spread_from_volatility(volatility, base_spread, vol_coefficient)
        financing_data = self.calculate_financing_cost(notional=notional, days=days, spread=spread, sofr_rate=sofr_rate)

        cash_yield = dividend_yield * (1 - tax_rate)
        synthetic_yield = dividend_yield
        advantage = synthetic_yield - cash_yield
        annual_advantage = advantage * notional
        position_advantage = annual_advantage * (days / 365)

        net_cost = financing_data['total_cost'] - position_advantage

        print(f"\nDividend Impact:")
        print(f"  Yield: {dividend_yield:.2%} → Cash: {cash_yield:.2%} → Synth: {synthetic_yield:.2%}")
        print(f"  Advantage: {advantage:.2%} (${position_advantage:.0f} for position)")

        print(f"\nNet Cost:")
        print(f"  Gross: ${financing_data['total_cost']:.0f} - Dividend: ${position_advantage:.0f}")
        print(f"  Net: ${net_cost:.0f}")

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
                current_date = dividends.index[-1] if not dividends.empty else pd.Timestamp.now()
                one_year_ago = current_date - pd.DateOffset(years=1)
                recent_dividends = dividend_series[dividend_series.index >= one_year_ago]
                annual_dividend = recent_dividends.sum()
                dividend_yield = annual_dividend / price if price > 0 else 0
        dividend_yield = dividend_yield or 0.0

        print(f"\nParameters:")
        print(f"  Ticker: {ticker} | Price: ${price:.2f} | Notional: ${notional:,.0f}")
        print(f"  Vol: {volatility:.1%} | Div Yield: {dividend_yield:.2%} | Days: {days}")

        # Synthetic costs
        spread = self.estimate_spread_from_volatility(volatility)
        synthetic_rate = sofr_rate + spread
        print("Syntethic", end=' ')
        financing_data = self.calculate_financing_cost(notional, days, spread, sofr_rate)

        cash_yield = dividend_yield * (1 - tax_rate)
        synthetic_yield = dividend_yield
        dividend_advantage = synthetic_yield - cash_yield
        annual_advantage = dividend_advantage * notional
        period_advantage = annual_advantage * (days / 365)

        transaction_cost_amount = transaction_cost * notional
        total_synthetic_cost = financing_data['total_cost'] - period_advantage + transaction_cost_amount

        # Cash costs
        print("Long Cash", end=' ')
        opportunity_cost = self.calculate_financing_cost(notional, days, 0, sofr_rate)['total_cost']
        print("Short Cash", end=' ')
        borrow_cost = self.calculate_financing_cost(notional, days, 0, cash_borrow_rate)['total_cost']
        total_cash_long = opportunity_cost + transaction_cost_amount
        total_cash_short = borrow_cost + transaction_cost_amount

        # Comparison
        savings_vs_long = total_cash_long - total_synthetic_cost
        savings_vs_short = total_cash_short - total_synthetic_cost

        costs = [('Synthetic', total_synthetic_cost),
                 ('Cash Long', total_cash_long),
                 ('Cash Short', total_cash_short)]
        cheapest = min(costs, key=lambda x: x[1])

        print(f"\nCost Comparison:")
        for name, cost in costs:
            print(f"  {name}: ${cost:,.0f}")
        print(f"\nCheapest: {cheapest[0]} (${cheapest[1]:,.0f})")

        return {
            'ticker': ticker,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'price': price,
            'volatility': volatility,
            'dividend_yield': dividend_yield,
            'estimated_spread': spread,
            'synthetic_rate': synthetic_rate,
            'synthetic_cost': {
                'financing_cost': financing_data['total_cost'],
                'dividend_advantage': period_advantage,
                'transaction_cost': transaction_cost_amount,
                'total': total_synthetic_cost
            },
            'cash_cost': {
                'long': {'total': total_cash_long},
                'short': {'total': total_cash_short}
            },
            'comparison': {
                'synthetic_vs_long': savings_vs_long,
                'synthetic_vs_short': savings_vs_short,
                'recommendation': cheapest[0].upper().replace(' ', '_')
            }
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
                    print(f"\n{ticker}: ", end="")
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
                    print(f" Done (${result['synthetic_cost']['total']:.0f})")

            except Exception as e:
                print(f"Error with {ticker}: {str(e)[:50]}")

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Summary statistics
        if 'comparison' in results_df.columns:
            recs = results_df['comparison'].apply(lambda x: x['recommendation'])
            print(f"\nRecommendations:")
            for rec, count in recs.value_counts().items():
                print(f"  {rec}: {count} tickers")

        if 'synthetic_cost' in results_df.columns:
            results_df['synthetic_total'] = results_df['synthetic_cost'].apply(lambda x: x['total'])
            print(f"\nMost Expensive Synthetic:")
            for _, row in results_df.nlargest(3, 'synthetic_total').iterrows():
                print(f"  {row['ticker']}: ${row['synthetic_total']:.0f}")

        return results_df

    def calculate_basis(self,
                        synthetic_rate: float,
                        cash_rate: float) -> float:
        """Calculate basis between synthetic and cash rates"""
        basis = synthetic_rate - cash_rate
        bps = basis * 10000

        print(f"\nBasis Calculation:")
        print(f"  Synthetic: {synthetic_rate:.2%} | Cash: {cash_rate:.2%}")
        print(f"  Basis: {basis:.2%} ({bps:.0f} bps)")

        if basis > 0:
            print(f"  → Synthetic is {basis:.2%} more expensive")
        elif basis < 0:
            print(f"  → Synthetic is {-basis:.2%} cheaper")

        return basis

    def calculate_historical_basis(self,
                                   tickers: List[str],
                                   prices: pd.DataFrame,
                                   volatilities: pd.DataFrame,
                                   dividends: Optional[pd.DataFrame] = None,
                                   sofr_rates: Optional[pd.Series] = None,
                                   cash_rates: Optional[pd.Series] = None,
                                   days: int = 90,
                                   notional: float = 100000,
                                   tax_rate: float = 0.30) -> pd.DataFrame:
        """Calculate historical basis over time"""
        print(f"Historical Basis for {len(tickers)} tickers")
        print(f"Date Range: {prices.index[0].date()} to {prices.index[-1].date()}")

        basis_data = []
        for ticker in tickers:
            if ticker not in prices.columns or ticker not in volatilities.columns:
                continue

            print(f".", end="")  # Progress indicator
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
                        'dividend_yield': div_yield
                    })
                except:
                    continue

        if not basis_data:
            return pd.DataFrame()

        basis_df = pd.DataFrame(basis_data)
        pivot_basis = basis_df.pivot(index='date', columns='ticker', values='basis')

        print(f"\n\nBasis Statistics:")
        for ticker in pivot_basis.columns:
            basis_series = pivot_basis[ticker].dropna()
            if len(basis_series) > 0:
                print(f"  {ticker}: Mean: {basis_series.mean():.2%} | "
                      f"Std: {basis_series.std():.2%} | Current: {basis_series.iloc[-1]:.2%}")

        basis_file = "data/processed/historical_basis.csv"
        basis_df.to_csv(basis_file, index=False)
        print(f"\nSaved to: {basis_file}")

        return basis_df

    def generate_basis_signals(self,
                               basis_series: pd.Series,
                               entry_threshold: float = 0.01,
                               exit_threshold: float = 0.0025,
                               lookback_days: int = 252) -> pd.DataFrame:
        """Generate trading signals based on basis levels"""
        print(f"\nBasis Signals:")
        print(f"  Entry: ±{entry_threshold:.2%} | Exit: ±{exit_threshold:.2%}")

        signals = pd.DataFrame(index=basis_series.index)
        signals['basis'] = basis_series
        signals['mean'] = basis_series.rolling(lookback_days, min_periods=20).mean()
        signals['std'] = basis_series.rolling(lookback_days, min_periods=20).std()
        signals['zscore'] = (basis_series - signals['mean']) / signals['std']

        signals['signal'] = 0
        signals.loc[basis_series < -entry_threshold, 'signal'] = 1  # Long basis
        signals.loc[basis_series > entry_threshold, 'signal'] = -1  # Short basis
        signals.loc[basis_series.abs() < exit_threshold, 'signal'] = 0
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)

        print(f"\nSignal Summary:")
        for pos, label in [(1, 'Long'), (-1, 'Short'), (0, 'Neutral')]:
            count = (signals['position'] == pos).sum()
            print(f"  {label}: {count} days")

        if len(signals) > 0:
            current = signals.iloc[-1]
            if current['position'] == 1:
                print(f"  Current: LONG BASIS (Synthetic is cheap by {-current['basis']:.2%})")
            elif current['position'] == -1:
                print(f"  Current: SHORT BASIS (Synthetic is expensive by {current['basis']:.2%})")
            else:
                print(f"  Current: NEUTRAL")

        return signals

    def analyze_basis_regimes(self,
                              basis_df: pd.DataFrame,
                              ticker: str) -> Dict[str, Any]:
        """Analyze different basis regimes for a specific ticker"""
        print(f"\nBasis Regimes: {ticker}")

        ticker_data = basis_df[basis_df['ticker'] == ticker].copy()
        if ticker_data.empty:
            return {}

        basis_values = ticker_data['basis'].dropna()
        low_pct = basis_values.quantile(0.25)
        high_pct = basis_values.quantile(0.75)

        regimes = []
        current_regime = None
        regime_start = None

        for i, (date, row) in enumerate(ticker_data.iterrows()):
            basis = row['basis']
            if basis < low_pct:
                regime = 'CHEAP'
            elif basis > high_pct:
                regime = 'EXPENSIVE'
            else:
                regime = 'NORMAL'

            if regime != current_regime:
                if current_regime is not None:
                    regimes.append({
                        'regime': current_regime,
                        'start': regime_start,
                        'end': ticker_data.iloc[i - 1]['date'],
                        'duration': (ticker_data.iloc[i - 1]['date'] - regime_start).days,
                        'avg_basis': ticker_data.loc[regime_start:ticker_data.iloc[i - 1]['date'], 'basis'].mean()
                    })
                current_regime = regime
                regime_start = date

        # Add last regime
        if current_regime is not None:
            regimes.append({
                'regime': current_regime,
                'start': regime_start,
                'end': ticker_data.iloc[-1]['date'],
                'duration': (ticker_data.iloc[-1]['date'] - regime_start).days,
                'avg_basis': ticker_data.loc[regime_start:, 'basis'].mean()
            })

        # Statistics
        regime_stats = {}
        for regime_type in ['CHEAP', 'EXPENSIVE', 'NORMAL']:
            type_regimes = [r for r in regimes if r['regime'] == regime_type]
            if type_regimes:
                total_days = sum(r['duration'] for r in type_regimes)
                avg_basis = sum(r['avg_basis'] * r['duration'] for r in type_regimes) / total_days

                regime_stats[regime_type] = {
                    'count': len(type_regimes),
                    'total_days': total_days,
                    'avg_duration': total_days / len(type_regimes),
                    'avg_basis': avg_basis
                }

        print(f"\nRegime Analysis:")
        for regime, stats in regime_stats.items():
            print(f"  {regime}: {stats['count']} regimes, Avg {stats['avg_duration']:.0f} days, "
                  f"Basis: {stats['avg_basis']:.2%}")

        current_basis = ticker_data['basis'].iloc[-1]
        current_regime = 'CHEAP' if current_basis < low_pct else 'EXPENSIVE' if current_basis > high_pct else 'NORMAL'
        print(f"\nCurrent: {current_regime} ({current_basis:.2%})")

        return {
            'regime_data': regimes,
            'regime_stats': regime_stats,
            'current_regime': current_regime,
            'current_basis': current_basis,
            'percentiles': {'25th': low_pct, '75th': high_pct}
        }