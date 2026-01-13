#!/usr/bin/env python3
"""
Synthetic Spread Arbitrage Optimizer
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import DataPipeline
from src.synthetic_pricer import SyntheticPricer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Synthetic Spread Arbitrage Optimizer")
    parser.add_argument('--tickers', nargs='+', help='Stock tickers to analyze')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--notional', type=float, default=100000, help='Position size in dollars')
    parser.add_argument('--days', type=int, default=90, help='Holding period in days')
    parser.add_argument('--quick', action='store_true', help='Quick mode for testing')
    return parser.parse_args()


def run_data_pipeline(tickers=None, start=None, end=None, quick=False):
    """Run the data pipeline to fetch and process market data"""
    print("Data Pipeline")
    print("-" * 50)

    pipeline = DataPipeline()

    # Override defaults if provided
    if tickers:
        pipeline.tickers = tickers
    if quick:
        pipeline.tickers = pipeline.tickers[:3]
        start = '2024-01-01'
    if start:
        pipeline.start_date = start
    if end:
        pipeline.end_date = end

    print("Fetching market data...")
    prices = pipeline.fetch_market_data()

    print("Calculating returns...")
    returns = pipeline.calculate_returns(prices)

    print("Calculating volatility...")
    volatility = pipeline.calculate_realized_volatility(returns, window=30)

    print("Fetching dividends...")
    dividends = pipeline.get_dividend_data()

    return prices, returns, volatility, dividends


def run_analysis(prices, volatility, dividends, notional=100000, days=90):
    """Run synthetic pricer analysis"""
    print("\nSynthetic Pricer")
    print("-" * 50)

    pricer = SyntheticPricer()

    print("Financing cost example:")
    pricer.calculate_financing_cost(notional=notional, days=days)

    print("\nBatch cost analysis...")
    results = pricer.batch_cost_analysis(
        tickers=prices.columns.tolist(),
        prices=prices,
        volatilities=volatility,
        dividends=dividends if not dividends.empty else None,
        days=days,
        notional=notional,
        tax_rate=0.30
    )

    if not results.empty:
        print_results(results)

    return results


def print_results(results):
    """Print analysis results"""
    print(f"\nResults for {len(results)} tickers:")

    # Count recommendations
    rec_counts = results['comparison'].apply(lambda x: x['recommendation']).value_counts()
    print("\nRecommendation Summary:")
    for rec, count in rec_counts.items():
        print(f"  {rec.replace('_', ' ').title()}: {count} tickers")

    # Show top 5 by savings
    results['savings'] = results['comparison'].apply(lambda x: x.get('savings', x.get('synthetic_vs_long', 0)))
    top_savings = results.nlargest(5, 'savings')

    print("\nTop 5 Savings (vs Long Cash):")
    for _, row in top_savings.iterrows():
        print(f"  {row['ticker']}: ${row['savings']:.0f}")


def save_results(results):
    """Save results to file"""
    if results.empty:
        print("No results to save")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs('data/results', exist_ok=True)

    filename = f'data/results/results_{timestamp}.csv'
    results.to_csv(filename, index=False)
    print(f"\nResults saved: {filename}")

    return filename


def main():
    """Main execution function"""
    args = parse_args()

    print("=" * 60)
    print("Synthetic Spread Arbitrage Optimizer")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    # Run data pipeline
    prices, returns, volatility, dividends = run_data_pipeline(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        quick=args.quick
    )

    # Run analysis
    results = run_analysis(
        prices=prices,
        volatility=volatility,
        dividends=dividends,
        notional=args.notional,
        days=args.days
    )

    # Save results
    save_results(results)

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()