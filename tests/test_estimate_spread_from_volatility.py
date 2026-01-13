# test_2_2.py
"""
Test Component 2.2: estimate_spread_from_volatility()
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic_pricer import SyntheticPricer


def test_estimate_spread():
    """Test the spread estimation from volatility"""

    print("Testing Component 2.2: estimate_spread_from_volatility()")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Test 1: Single value example from documentation
    print("\nTest 1: Single value (AAPL with 25% volatility)")
    print("-" * 60)

    # Example from documentation:
    # Base_Spread = 15 bps (0.0015)
    # Vol_Coefficient = 20 bps per vol point (0.002)
    # For AAPL with 25% volatility (0.25):
    # Estimated_Spread = 0.0015 + (0.002 × 0.25) = 0.0020 (20 bps)

    aapl_vol = 0.25  # 25%
    spread = pricer.estimate_spread_from_volatility(
        volatility=aapl_vol,
        base_spread=0.0015,
        vol_coefficient=0.002
    )

    expected_spread = 0.0015 + (0.002 * 0.25)
    print(f"\nExpected: {expected_spread:.4f} ({expected_spread * 100:.2f}%)")
    print(f"Actual: {spread:.4f} ({spread * 100:.2f}%)")

    if abs(spread - expected_spread) < 0.0001:
        print("Calculation matches")
    else:
        print("Calculation doesn't match")

    # Test 2: Series input
    print("\nTest 2: Series input (multiple volatilities)")
    print("-" * 60)

    vols_series = pd.Series([0.20, 0.25, 0.30, 0.35, 0.40],
                            index=['Day1', 'Day2', 'Day3', 'Day4', 'Day5'])

    spreads_series = pricer.estimate_spread_from_volatility(
        volatility=vols_series,
        base_spread=0.0015,
        vol_coefficient=0.002
    )

    print(f"\nVolatility series:")
    for day, vol in vols_series.items():
        print(f"  {day}: {vol * 100:.1f}%")

    print(f"\nEstimated spreads:")
    for day, spread_val in spreads_series.items():
        print(f"  {day}: {spread_val * 100:.2f}%")

    # Test 3: DataFrame input
    print("\nTest 3: DataFrame input (multiple stocks)")
    print("-" * 60)

    # Create sample volatility DataFrame
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    vol_df = pd.DataFrame({
        'AAPL': [0.22, 0.23, 0.25, 0.24, 0.26],
        'MSFT': [0.18, 0.19, 0.20, 0.21, 0.22],
        'TSLA': [0.45, 0.47, 0.50, 0.48, 0.52],
        'NVDA': [0.35, 0.38, 0.40, 0.42, 0.45]
    }, index=dates)

    spread_df = pricer.estimate_spread_from_volatility(
        volatility=vol_df,
        base_spread=0.0015,
        vol_coefficient=0.002
    )

    print(f"\nVolatility DataFrame (first 3 rows):")
    print(vol_df.head(3))

    print(f"\nEstimated Spread DataFrame (first 3 rows):")
    print(spread_df.head(3))

    # Test 4: Calculate statistics
    print("\nTest 4: Spread statistics")
    print("-" * 60)

    stats = pricer.calculate_spread_statistics(vol_df)

    # Test 5: Real-world comparison
    print("\nTest 5: Real-world comparison")
    print("-" * 60)

    print("\nComparing AAPL (low vol) vs TSLA (high vol):")
    aapl_vol = 0.25  # 25%
    tsla_vol = 0.50  # 50%

    aapl_spread = pricer.estimate_spread_from_volatility(
        volatility=aapl_vol,
        base_spread=0.0015,
        vol_coefficient=0.002
    )

    tsla_spread = pricer.estimate_spread_from_volatility(
        volatility=tsla_vol,
        base_spread=0.0015,
        vol_coefficient=0.002
    )

    print(f"\nAAPL (vol {aapl_vol * 100:.0f}%):")
    print(f"  Spread = 0.0015 + (0.002 × {aapl_vol})")
    print(f"  Spread = 0.0015 + {0.002 * aapl_vol:.4f}")
    print(f"  Spread = {aapl_spread:.4f} = {aapl_spread * 100:.2f}%")

    print(f"\nTSLA (vol {tsla_vol * 100:.0f}%):")
    print(f"  Spread = 0.0015 + (0.002 × {tsla_vol})")
    print(f"  Spread = 0.0015 + {0.002 * tsla_vol:.4f}")
    print(f"  Spread = {tsla_spread:.4f} = {tsla_spread * 100:.2f}%")

    print(f"\nSpread difference: {(tsla_spread - aapl_spread) * 100:.2f}%")

    # Test 6: With different coefficients
    print("\nTest 6: Different volatility coefficients")
    print("-" * 60)

    for coefficient in [0.001, 0.002, 0.003]:
        spread = pricer.estimate_spread_from_volatility(
            volatility=0.30,  # 30% volatility
            base_spread=0.0015,
            vol_coefficient=coefficient
        )
        print(f"  Coefficient {coefficient * 100:.1f} bps/vol: spread = {spread * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Component 2.2 tests complete!")

    return spread_df, stats


def test_integration_with_data_pipeline():
    """Test integration with data pipeline components"""

    print("\nTesting integration with Data Pipeline")
    print("=" * 60)

    try:
        # Import data pipeline
        from src.data_pipeline import DataPipeline

        # Create data pipeline
        data_pipe = DataPipeline()

        # Fetch market data
        print("Fetching market data...")
        prices = data_pipe.fetch_market_data(
            tickers=['AAPL', 'MSFT', 'TSLA'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Calculate returns
        print("Calculating returns...")
        returns = data_pipe.calculate_returns(prices)

        # Calculate volatility
        print("Calculating volatility...")
        volatility = data_pipe.calculate_realized_volatility(returns, window=30)

        # Create synthetic pricer
        pricer = SyntheticPricer()

        # Estimate spreads from volatility
        print("\nEstimating spreads from volatility...")
        spreads = pricer.estimate_spread_from_volatility(volatility)

        print(f"\nCurrent spreads (latest date):")
        latest_spreads = spreads.iloc[-1]
        for ticker, spread in latest_spreads.items():
            print(f"  {ticker}: {spread * 100:.2f}%")

        # Calculate financing costs
        print("\nCalculating financing costs...")
        notionals = {
            'AAPL': 100000,
            'MSFT': 100000,
            'TSLA': 100000
        }

        financing_costs = pricer.calculate_multiple_positions(notionals, days=90)

        return volatility, spreads, financing_costs

    except ImportError as e:
        print(f"Could not import DataPipeline: {e}")
        return None, None, None