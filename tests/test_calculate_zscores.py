import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from statistical_models import StatisticalModels


def create_test_spreads():
    """Create synthetic spread data with some extreme values"""
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=100, freq='B')

    # Create spreads with some extreme values
    spreads = []
    for i in range(100):
        if i == 80:  # One extreme high value
            spreads.append(0.0050)  # 50 bps
        elif i == 85:  # One extreme low value
            spreads.append(0.0005)  # 5 bps
        else:
            spreads.append(0.0020 + np.random.normal(0, 0.0003))  # ~20 bps

    return pd.Series(spreads, index=dates, name='TEST_SPREAD')


def test_basic_zscore():
    """Test basic Z-score calculation"""
    print("Test: Basic Z-Score Calculation")
    print("=" * 50)

    # Create test data
    spreads = create_test_spreads()

    print(f"Spread data: {len(spreads)} points")
    print(f"Mean: {spreads.mean():.6f}, Std: {spreads.std():.6f}")

    # Initialize model
    model = StatisticalModels(lookback_days=60)

    # Calculate Z-scores
    zscores = model.calculate_zscore(spreads)

    print(f"\nZ-score range: [{zscores.min():.2f}, {zscores.max():.2f}]")

    # Analyze extreme points
    print("\nAnalyzing extreme points:")
    for i, (date, spread) in enumerate(spreads.items()):
        zscore = zscores.loc[date]
        if abs(zscore) > 2.0:
            print(f"  {date.date()}: Spread={spread:.6f}, Z={zscore:.2f}")

    # Test interpretation
    print("\nZ-score interpretations (last 5 days):")
    for i in range(-5, 0):
        date = spreads.index[i]
        spread = spreads.iloc[i]
        zscore = zscores.iloc[i]

        interpretation = model.interpret_zscore(zscore, threshold=2.0)

        print(f"\n  {date.date()}:")
        print(f"    Spread: {spread:.6f}")
        print(f"    Z-score: {zscore:.2f}")
        print(f"    Signal: {interpretation['signal']}")
        print(f"    Interpretation: {interpretation['interpretation']}")
        print(f"    Probability extreme: {interpretation['prob_extreme']:.1%}")

    return zscores


def test_multiple_series():
    """Test Z-score calculation for multiple series"""
    print("\n\nTest: Multiple Series Z-Scores")
    print("=" * 50)

    dates = pd.date_range('2024-01-01', periods=100, freq='B')

    # Create DataFrame with multiple spread series
    spreads_df = pd.DataFrame({
        'AAPL': 0.0020 + np.random.normal(0, 0.0002, 100),
        'MSFT': 0.0015 + np.random.normal(0, 0.0003, 100),
        'GOOGL': 0.0025 + np.random.normal(0, 0.0004, 100)
    }, index=dates)

    # Add some extreme values
    spreads_df.loc[dates[80], 'AAPL'] = 0.0050  # Extreme high
    spreads_df.loc[dates[85], 'MSFT'] = 0.0003  # Extreme low

    # Initialize model
    model = StatisticalModels(lookback_days=60)

    # Calculate Z-scores for all series
    zscores_df = model.calculate_zscore(spreads_df)

    print(f"\nCurrent Z-scores:")
    for ticker in spreads_df.columns:
        current_spread = spreads_df[ticker].iloc[-1]
        current_z = zscores_df[ticker].iloc[-1]
        interpretation = model.interpret_zscore(current_z, threshold=2.0)

        print(f"\n  {ticker}:")
        print(f"    Spread: {current_spread:.6f}")
        print(f"    Z-score: {current_z:.2f}")
        print(f"    Signal: {interpretation['signal']}")
        print(f"    Significance: {interpretation['significance']}")

    return zscores_df


def test_with_realistic_basis():
    """Test with realistic basis data"""
    print("\n\nTest: Realistic Basis Analysis")
    print("=" * 50)

    dates = pd.date_range('2023-01-01', periods=252, freq='B')  # 1 year

    # Create realistic basis data (synthetic vs cash spread)
    basis_values = []
    current = 0.0020  # 20 bps

    for i in range(len(dates)):
        # Add some mean reversion and randomness
        reversion = 0.1 * (0.0020 - current)  # Mean reversion to 20 bps
        noise = np.random.normal(0, 0.0005)
        current = current + reversion + noise
        basis_values.append(current)

    basis_series = pd.Series(basis_values, index=dates, name='BASIS')

    print(f"Basis Statistics:")
    print(f"  Mean: {basis_series.mean():.6f} ({basis_series.mean() * 10000:.1f} bps)")
    print(f"  Std: {basis_series.std():.6f} ({basis_series.std() * 10000:.1f} bps)")
    print(f"  Min: {basis_series.min():.6f} ({basis_series.min() * 10000:.1f} bps)")
    print(f"  Max: {basis_series.max():.6f} ({basis_series.max() * 10000:.1f} bps)")

    # Calculate Z-scores
    model = StatisticalModels(lookback_days=126)  # 6-month lookback
    zscores = model.calculate_zscore(basis_series)

    # Find extreme periods
    extreme_mask = zscores.abs() > 2.0
    extreme_dates = zscores[extreme_mask].index

    print(f"\nExtreme Basis Periods (|Z| > 2.0):")
    print(f"  Found {len(extreme_dates)} extreme days ({len(extreme_dates) / len(dates):.1%} of time)")

    if len(extreme_dates) > 0:
        print("\n  Sample extreme periods:")
        for date in extreme_dates[:5]:  # Show first 5
            basis = basis_series.loc[date]
            zscore = zscores.loc[date]
            interpretation = model.interpret_zscore(zscore, threshold=2.0)

            print(f"    {date.date()}: Basis={basis:.6f} ({basis * 10000:.1f} bps), "
                  f"Z={zscore:.2f}, {interpretation['signal']}")

    # Current analysis
    current_basis = basis_series.iloc[-1]
    current_z = zscores.iloc[-1]
    interpretation = model.interpret_zscore(current_z, threshold=2.0)

    print(f"\nCurrent Analysis:")
    print(f"  Basis: {current_basis:.6f} ({current_basis * 10000:.1f} bps)")
    print(f"  Z-score: {current_z:.2f}")
    print(f"  Signal: {interpretation['signal']}")
    print(f"  Interpretation: {interpretation['interpretation']}")

    return basis_series, zscores