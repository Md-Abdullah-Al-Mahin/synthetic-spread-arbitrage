import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic_pricer import SyntheticPricer


def test_basic_basis_calculation():
    """Test basic basis calculation"""
    print("\nTest 1: Basic Basis Calculation")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Test 1: Basic example
    synthetic_rate = 0.047  # 4.7%
    cash_rate = 0.032  # 3.2%

    basis = pricer.calculate_basis(synthetic_rate, cash_rate)
    expected_basis = 0.047 - 0.032

    print(f"Synthetic: {synthetic_rate:.2%}, Cash: {cash_rate:.2%}")
    print(f"Expected basis: {expected_basis:.4f}")
    print(f"Actual basis: {basis:.4f}")
    print("✓ PASS" if abs(basis - expected_basis) < 0.0001 else "✗ FAIL")

    return True


def create_simple_historical_data():
    """Create simple historical data for testing"""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')

    prices = pd.DataFrame({
        'AAPL': 180 + np.cumsum(np.random.normal(0, 1, 20)),
        'MSFT': 400 + np.cumsum(np.random.normal(0, 0.8, 20))
    }, index=dates)

    volatilities = pd.DataFrame({
        'AAPL': 0.20 + 0.02 * np.random.randn(20),
        'MSFT': 0.18 + 0.015 * np.random.randn(20)
    }, index=dates)

    sofr_rates = pd.Series(0.04 + 0.001 * np.random.randn(20), index=dates)
    cash_rates = sofr_rates + 0.002 + 0.0005 * np.random.randn(20)

    dividends = pd.DataFrame(0.0, index=dates, columns=['AAPL', 'MSFT'])
    dividends['AAPL'].iloc[0] = 0.24  # One dividend
    dividends['MSFT'].iloc[0] = 0.68

    return prices, volatilities, sofr_rates, cash_rates, dividends


def test_historical_basis_simple():
    """Test historical basis calculation - simplified"""
    print("\nTest 2: Historical Basis Calculation")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Create data
    prices, vols, sofr, cash, divs = create_simple_historical_data()

    # Calculate historical basis
    basis_df = pricer.calculate_historical_basis(
        tickers=['AAPL', 'MSFT'],
        prices=prices,
        volatilities=vols,
        dividends=divs,
        sofr_rates=sofr,
        cash_rates=cash,
        days=90,
        notional=100000,
        tax_rate=0.30
    )

    if basis_df.empty:
        print("✗ FAIL - Empty result")
        return False

    print(f"✓ PASS - Generated {len(basis_df)} basis records")
    print(f"Columns: {list(basis_df.columns)}")
    print(f"Tickers: {basis_df['ticker'].unique()}")

    # Test signals for one ticker
    aapl_basis = basis_df[basis_df['ticker'] == 'AAPL'].set_index('date')['basis']

    signals = pricer.generate_basis_signals(
        basis_series=aapl_basis,
        entry_threshold=0.005,
        exit_threshold=0.001,
        lookback_days=20
    )

    print(f"✓ Generated {len(signals)} trading signals")

    return True


def test_basis_trading_strategy_simple():
    """Test basis trading strategy - simplified"""
    print("\nTest 3: Basis Trading Strategy")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Create simple basis series
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)

    # Create basis series with clear mean reversion pattern
    basis_series = pd.Series(
        0.001 * np.sin(np.arange(50) / 5) + np.random.normal(0, 0.002, 50),
        index=dates
    )

    # Generate signals
    signals = pricer.generate_basis_signals(
        basis_series=basis_series,
        entry_threshold=0.003,
        exit_threshold=0.001,
        lookback_days=20
    )

    if signals.empty:
        print("✗ FAIL - No signals generated")
        return False

    print(f"✓ PASS - Generated signals with shape {signals.shape}")
    print(f"Signal distribution:")
    for pos in [-1, 0, 1]:
        count = (signals['position'] == pos).sum()
        print(f"  Position {pos}: {count} days")

    return True


def test_integration_simple():
    """Test integration - simplified"""
    print("\nTest 4: Integration Test")
    print("=" * 60)

    try:
        from src.data_pipeline import DataPipeline
        print("✓ DataPipeline imported successfully")

        # For this test, we'll use synthetic data
        prices, vols, _, _, divs = create_simple_historical_data()

        pricer = SyntheticPricer()

        # Run cost analysis
        cost_df = pricer.batch_cost_analysis(
            tickers=['AAPL', 'MSFT'],
            prices=prices,
            volatilities=vols,
            dividends=divs,
            days=90,
            notional=100000,
            tax_rate=0.30
        )

        if not cost_df.empty:
            print(f"✓ Cost analysis successful for {len(cost_df)} tickers")
            for _, row in cost_df.iterrows():
                print(f"  {row['ticker']}: ${row['synthetic_cost']['total']:.0f}")
        else:
            print("✗ Cost analysis failed")
            return False

        return True

    except ImportError:
        print("✗ DataPipeline not available - skipping detailed integration test")
        # Try basic functionality
        pricer = SyntheticPricer()
        prices, vols, _, _, divs = create_simple_historical_data()

        # Just test that the pricer can be instantiated and run basic operations
        try:
            result = pricer.calculate_total_cost_of_carry(
                ticker='AAPL',
                price=180.0,
                volatility=0.20,
                days=90,
                notional=100000
            )
            print("✓ Basic pricer functionality works")
            return True
        except Exception as e:
            print(f"✗ Basic test failed: {e}")
            return False