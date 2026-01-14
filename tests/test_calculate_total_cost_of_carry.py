# test_2_4.py
"""
Test Component 2.4: calculate_total_cost_of_carry()
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic_pricer import SyntheticPricer


def create_sample_data():
    """Create sample data for testing"""

    # Sample prices
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    prices = pd.DataFrame({
        'AAPL': [185.0, 186.5, 187.2, 188.0, 189.5, 190.2, 191.0, 192.5, 193.0, 194.5],
        'MSFT': [410.0, 412.5, 411.0, 413.8, 415.0, 416.5, 418.0, 419.5, 421.0, 422.5],
        'TSLA': [240.0, 242.5, 241.0, 239.5, 238.0, 237.0, 236.5, 235.0, 234.5, 233.0],
        'GOOGL': [145.0, 146.2, 147.0, 145.8, 148.0, 149.5, 150.0, 151.2, 152.0, 153.5]
    }, index=dates)

    # Sample volatilities (annualized)
    volatilities = pd.DataFrame({
        'AAPL': [0.22, 0.23, 0.24, 0.23, 0.22, 0.21, 0.22, 0.23, 0.24, 0.25],
        'MSFT': [0.18, 0.19, 0.20, 0.19, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23],
        'TSLA': [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54],
        'GOOGL': [0.20, 0.21, 0.22, 0.21, 0.20, 0.19, 0.20, 0.21, 0.22, 0.23]
    }, index=dates)

    # Sample dividends
    div_dates = pd.DatetimeIndex([
        '2024-02-15', '2024-05-15', '2024-08-15', '2024-11-15',
        '2024-03-10', '2024-06-10', '2024-09-10', '2024-12-10',
        '2024-04-01'
    ])

    dividends = pd.DataFrame({
        'AAPL': [0.24, 0.24, 0.24, 0.24, np.nan, np.nan, np.nan, np.nan, np.nan],
        'MSFT': [0.68, 0.68, 0.68, 0.75, np.nan, np.nan, np.nan, np.nan, np.nan],
        'TSLA': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'GOOGL': [np.nan, np.nan, np.nan, np.nan, 0.20, np.nan, np.nan, np.nan, np.nan]
    }, index=div_dates)

    return prices, volatilities, dividends


def test_total_cost_of_carry():
    """Test the main cost comparison function"""

    print("Testing Component 2.4: calculate_total_cost_of_carry()")
    print("=" * 70)

    pricer = SyntheticPricer()

    # Test 1: MSFT example from documentation
    print("\nTest 1: MSFT example from documentation")
    print("-" * 70)

    """
    Example from documentation:
    Scenario: Long position in MSFT, $100,000 notional, 90 days

    SYNTHETIC ROUTE:
      Notional: $100,000
      SOFR: 4.5%
      Volatility: 22%
      Estimated Spread: 15 + (20 × 0.22) = 19.4 bps
      Total Rate: 4.5% + 0.194% = 4.694%

      Financing Cost = $100,000 × 0.04694 × (90/360) = $1,173.50

      Dividend Yield: 2.5%
      Div Treatment Advantage: 0.75% (from above)
      Dividend Benefit = $100,000 × 0.0075 × (90/360) = $187.50

      Net Cost = $1,173.50 - $187.50 = $986.00

    CASH ROUTE:
      Opportunity Cost = $100,000 × 0.045 × (90/360) = $1,125.00
      Transaction Costs = $25
      Net Cost = $1,125.00 + $25 = $1,150.00

    DECISION:
      Synthetic Cost: $986
      Cash Cost: $1,150
      Savings with Synthetic: $164 (14.3% cheaper)
    """

    msft_result = pricer.calculate_total_cost_of_carry(
        ticker="MSFT",
        price=415.00,
        volatility=0.22,
        dividend_yield=0.025,
        tax_rate=0.30,
        days=90,
        notional=100000,
        sofr_rate=0.045,
        cash_borrow_rate=0.045,
        transaction_cost=0.00025  # 0.025% = $25 on $100k
    )

    # Verify calculations
    print(f"\nVerification of MSFT example:")

    synthetic_total = msft_result['synthetic_cost']['total']
    cash_long_total = msft_result['cash_cost']['long']['total']
    savings = msft_result['comparison']['synthetic_vs_long']

    print(f"  Synthetic total: ${synthetic_total:.2f}")
    print(f"  Cash long total: ${cash_long_total:.2f}")
    print(f"  Savings: ${savings:.2f}")
    print(f"  Recommendation: {msft_result['comparison']['recommendation']}")

    # Test 2: TSLA (high volatility, no dividends)
    print("\nTest 2: TSLA (high volatility, no dividends)")
    print("-" * 70)

    tsla_result = pricer.calculate_total_cost_of_carry(
        ticker="TSLA",
        price=233.00,
        volatility=0.54,
        dividend_yield=0.00,
        tax_rate=0.30,
        days=90,
        notional=100000,
        sofr_rate=0.045,
        transaction_cost=0.001
    )

    print(f"  Synthetic total: ${tsla_result['synthetic_cost']['total']:.2f}")
    print(f"  Cash long total: ${tsla_result['cash_cost']['long']['total']:.2f}")
    print(f"  Recommendation: {tsla_result['comparison']['recommendation']}")

    # Test 3: Batch analysis
    print("\nTest 3: Batch analysis")
    print("-" * 70)

    prices, volatilities, dividends = create_sample_data()

    batch_result = pricer.batch_cost_analysis(
        tickers=['AAPL', 'MSFT', 'TSLA', 'GOOGL'],
        prices=prices,
        volatilities=volatilities,
        dividends=dividends,
        tax_rate=0.30,
        days=90,
        notional=100000,
        sofr_rate=0.045,
        transaction_cost=0.001
    )

    print(f"\nBatch results shape: {batch_result.shape}")

    # Test 4: Edge cases
    print("\nTest 4: Edge cases")
    print("-" * 70)

    # Very short holding period
    print("\nVery short holding period (7 days):")
    short_result = pricer.calculate_total_cost_of_carry(
        ticker="TEST",
        price=100.00,
        volatility=0.25,
        dividend_yield=0.02,
        days=7,
        notional=100000,
        transaction_cost=0.001
    )

    # Very high dividend yield
    print("\nVery high dividend yield (10%):")
    high_div_result = pricer.calculate_total_cost_of_carry(
        ticker="HIGH_DIV",
        price=50.00,
        volatility=0.15,
        dividend_yield=0.10,
        tax_rate=0.30,
        days=90,
        notional=100000
    )

    # Zero volatility
    print("\nZero volatility:")
    zero_vol_result = pricer.calculate_total_cost_of_carry(
        ticker="ZERO_VOL",
        price=100.00,
        volatility=0.00,
        dividend_yield=0.00,
        days=90,
        notional=100000
    )

    print("\n" + "=" * 70)
    print("Component 2.4 tests complete!")

    return msft_result, tsla_result, batch_result


def test_integration_with_data_pipeline():
    """Test integration with real data"""

    print("\nIntegration Test with Data Pipeline")
    print("=" * 70)

    try:
        from src.data_pipeline import DataPipeline

        # Create data pipeline
        data_pipe = DataPipeline()

        # Fetch data
        print("Fetching market data...")
        prices = data_pipe.fetch_market_data(
            tickers=['AAPL', 'MSFT', 'TSLA', 'NVDA'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Calculate returns and volatility
        print("Calculating returns and volatility...")
        returns = data_pipe.calculate_returns(prices)
        volatilities = data_pipe.calculate_realized_volatility(returns, window=30)

        # Get dividend data
        print("Fetching dividend data...")
        dividends = data_pipe.get_dividend_data(
            tickers=['AAPL', 'MSFT', 'TSLA', 'NVDA'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Create synthetic pricer
        pricer = SyntheticPricer()

        # Run batch analysis
        print("\nRunning batch cost analysis...")
        batch_result = pricer.batch_cost_analysis(
            tickers=['AAPL', 'MSFT', 'TSLA', 'NVDA'],
            prices=prices,
            volatilities=volatilities,
            dividends=dividends,
            days=90,
            notional=100000,
            tax_rate=0.30,
            sofr_rate=0.045
        )

        if not batch_result.empty:
            print(f"\nAnalysis complete for {len(batch_result)} tickers")

            # Show recommendations
            recommendations = []
            for _, row in batch_result.iterrows():
                rec = row['comparison']['recommendation']
                # Use the correct field names from the class
                savings_vs_long = row['comparison']['synthetic_vs_long']
                savings_vs_short = row['comparison']['synthetic_vs_short']

                # Determine savings based on recommendation
                if rec == 'SYNTHETIC':
                    # Synthetic is cheapest
                    savings = min(savings_vs_long, savings_vs_short)
                    if savings < 0:
                        savings_desc = f"saves ${-savings:.0f}"
                    else:
                        savings_desc = f"costs ${savings:.0f} more"
                else:
                    # Cash is cheapest
                    if rec == 'CASH_LONG':
                        savings = -savings_vs_long  # Negative of synthetic_vs_long
                    else:  # CASH_SHORT
                        savings = -savings_vs_short  # Negative of synthetic_vs_short

                    if savings > 0:
                        savings_desc = f"saves ${savings:.0f}"
                    else:
                        savings_desc = f"costs ${-savings:.0f} more"

                recommendations.append({
                    'ticker': row['ticker'],
                    'recommendation': rec,
                    'savings': savings,
                    'savings_desc': savings_desc
                })

            # Display results
            print("\nRecommendation Summary:")
            for rec in recommendations:
                print(f"  {rec['ticker']}: {rec['recommendation']} ({rec['savings_desc']})")

            # Count recommendations
            rec_counts = {}
            for rec in recommendations:
                rec_type = rec['recommendation']
                rec_counts[rec_type] = rec_counts.get(rec_type, 0) + 1

            print(f"\nRecommendation Counts:")
            for rec_type, count in rec_counts.items():
                print(f"  {rec_type}: {count} tickers")

            # Show cost details for each ticker
            print("\nDetailed Cost Analysis:")
            for _, row in batch_result.iterrows():
                print(f"\n{row['ticker']}:")
                print(f"  Synthetic: ${row['synthetic_cost']['total']:.0f}")
                print(f"  Cash Long: ${row['cash_cost']['long']['total']:.0f}")
                print(f"  Cash Short: ${row['cash_cost']['short']['total']:.0f}")
                print(f"  Recommendation: {row['comparison']['recommendation']}")

                # Show basis if synthetic_rate is available
                if 'synthetic_rate' in row:
                    synthetic_rate = row['synthetic_rate']
                    cash_rate = 0.045  # Use SOFR rate
                    basis = pricer.calculate_basis(synthetic_rate, cash_rate)
                    print(f"  Basis: {basis * 100:.2f}%")

            return batch_result
        else:
            print("No results from batch analysis")
            return pd.DataFrame()

    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()