#!/usr/bin/env python3
"""
Synthetic Spread Arbitrage Optimizer - Unified Interface
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import DataPipeline
from src.synthetic_pricer import SyntheticPricer
from src.statistical_models import StatisticalModels


def parse_args():
    """Simple command line arguments"""
    parser = argparse.ArgumentParser(description="Synthetic Spread Optimizer")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT'],
                        help='Stock tickers (default: AAPL MSFT)')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-06-01', help='End date')
    parser.add_argument('--stats', action='store_true', help='Run statistical analyses')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    return parser.parse_args()


def save_all_results(data, synthetic_results, stats_results, timestamp):
    """Save all results to files"""
    os.makedirs('data/results', exist_ok=True)

    # Save synthetic results
    if synthetic_results and 'batch_results' in synthetic_results:
        filename = f'data/results/synthetic_{timestamp}.csv'
        synthetic_results['batch_results'].to_csv(filename, index=False)
        print(f"  ✓ Synthetic results: {filename}")

    # Save statistical results
    if stats_results:
        # Save Z-scores
        if 'zscores' in stats_results:
            filename = f'data/results/zscores_{timestamp}.csv'
            stats_results['zscores'].to_csv(filename)
            print(f"  ✓ Z-scores: {filename}")

        # Save summary
        summary_file = f'data/results/summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Analysis Summary - {timestamp}\n")
            f.write("=" * 50 + "\n")

            if synthetic_results:
                f.write(f"\nSynthetic Analysis:\n")
                f.write(f"  Positions analyzed: {len(synthetic_results['batch_results'])}\n")
                f.write(f"  Average savings: ${synthetic_results['avg_savings']:.0f}\n")

            if stats_results and 'hypothesis' in stats_results:
                hyp = stats_results['hypothesis']
                f.write(f"\nStatistical Significance:\n")
                f.write(f"  t-statistic: {hyp.get('t_stat', 0):.2f}\n")
                f.write(f"  p-value: {hyp.get('p_value', 0):.4f}\n")
                f.write(f"  Significant: {hyp.get('reject_null', False)}\n")

        print(f"  ✓ Summary: {summary_file}")


def main():
    """Main function using unified interface"""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("=" * 60)
    print("SYNTHETIC SPREAD ARBITRAGE OPTIMIZER")
    print("=" * 60)

    # 1. Data Pipeline
    print("\n[1] DATA PIPELINE")
    print("-" * 40)
    pipeline = DataPipeline()

    if args.quick:
        args.tickers = args.tickers[:2]
        args.start = '2024-01-01'
        args.end = '2024-02-01'

    data = pipeline.run_complete_pipeline(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        include_vix=args.stats
    )

    # 2. Synthetic Analysis
    print("\n[2] SYNTHETIC PRICING")
    print("-" * 40)
    pricer = SyntheticPricer()

    synthetic_results = pricer.run_complete_analysis(
        prices=data['prices'],
        volatility=data['volatility'],
        dividends=data['dividends'],
        notional=100000,
        days=90
    )

    # 3. Statistical Analysis (if requested)
    stats_results = None
    if args.stats and 'spreads' in synthetic_results:
        print("\n[3] STATISTICAL ANALYSIS")
        print("-" * 40)

        model = StatisticalModels()

        # Get savings data for hypothesis test
        savings_data = None
        if 'batch_results' in synthetic_results:
            batch_df = synthetic_results['batch_results']
            if 'savings' in batch_df.columns:
                savings_data = batch_df['savings'].dropna()

        stats_results = model.run_complete_analysis(
            spreads=synthetic_results['spreads'],
            volatility=data['volatility'],
            returns=data['returns'],
            vix=data['vix'],
            savings_data=savings_data
        )

    # 4. Save Results
    print("\n[4] SAVING RESULTS")
    print("-" * 40)
    save_all_results(data, synthetic_results, stats_results, timestamp)

    # 5. Final Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    # Quick summary
    if synthetic_results:
        print(f"\nSynthetic Analysis Summary:")
        print(f"  Positions: {len(synthetic_results['batch_results'])}")
        print(f"  Avg Savings: ${synthetic_results['avg_savings']:.0f}")

    if stats_results and 'hypothesis' in stats_results:
        hyp = stats_results['hypothesis']
        if hyp.get('reject_null', False):
            print(f"\n✓ Statistically significant savings (p = {hyp.get('p_value', 0):.4f})")

    print(f"\nTime: {datetime.now().strftime('%H:%M:%S')}")
    print("Results saved to: data/results/")


if __name__ == "__main__":
    main()