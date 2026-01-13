import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic_pricer import SyntheticPricer


def test_basic_basis_calculation():
    """Test basic basis calculation"""

    print("Testing Component 2.5: calculate_basis()")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Test 1: Basic example from documentation
    print("\nTest 1: Basic example")
    print("-" * 60)

    synthetic_rate = 0.047  # 4.7%
    cash_rate = 0.032  # 3.2%

    basis = pricer.calculate_basis(synthetic_rate, cash_rate)

    expected_basis = 0.047 - 0.032  # 0.015 = 150 bps

    print(f"\nExpected basis: {expected_basis:.4f} ({expected_basis * 100:.2f}%)")
    print(f"Actual basis: {basis:.4f} ({basis * 100:.2f}%)")

    if abs(basis - expected_basis) < 0.0001:
        print("Calculation matches!")
    else:
        print("Calculation doesn't match")

    # Test 2: Negative basis (synthetic cheaper)
    print("\nTest 2: Negative basis")
    print("-" * 60)

    synthetic_rate2 = 0.038  # 3.8%
    cash_rate2 = 0.045  # 4.5%

    basis2 = pricer.calculate_basis(synthetic_rate2, cash_rate2)

    print(f"Interpretation: Synthetic is {abs(basis2) * 100:.2f}% CHEAPER than cash")

    # Test 3: Zero basis
    print("\nTest 3: Zero basis")
    print("-" * 60)

    basis3 = pricer.calculate_basis(0.042, 0.042)

    return basis, basis2, basis3


def create_historical_data():
    """Create historical data for testing"""

    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Create sample data
    np.random.seed(42)

    # Prices with trend and noise
    prices = pd.DataFrame({
        'AAPL': 180 + np.cumsum(np.random.normal(0, 1, 100)),
        'MSFT': 400 + np.cumsum(np.random.normal(0, 0.8, 100))
    }, index=dates)

    # Volatilities (mean-reverting)
    volatilities = pd.DataFrame({
        'AAPL': 0.20 + 0.05 * np.sin(np.arange(100) / 10) + np.random.normal(0, 0.02, 100),
        'MSFT': 0.18 + 0.04 * np.sin(np.arange(100) / 12) + np.random.normal(0, 0.015, 100)
    }, index=dates)

    # SOFR rates (slowly increasing)
    sofr_rates = pd.Series(
        0.040 + 0.001 * np.arange(100) / 100 + np.random.normal(0, 0.001, 100),
        index=dates
    )

    # Cash rates (slightly higher than SOFR)
    cash_rates = sofr_rates + 0.002 + np.random.normal(0, 0.0005, 100)

    # Dividends (quarterly)
    dividends = pd.DataFrame(index=dates, columns=['AAPL', 'MSFT'])
    for ticker in ['AAPL', 'MSFT']:
        div_series = pd.Series(0.0, index=dates)
        # Add quarterly dividends
        for i in range(0, 100, 20):
            if i < len(dates):
                div_series.iloc[i] = 0.24 if ticker == 'AAPL' else 0.68
        dividends[ticker] = div_series

    return prices, volatilities, sofr_rates, cash_rates, dividends


def test_historical_basis():
    """Test historical basis calculation"""

    print("\nTest: Historical basis calculation")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Create historical data
    print("Creating historical data...")
    prices, volatilities, sofr_rates, cash_rates, dividends = create_historical_data()

    print(f"Price data shape: {prices.shape}")
    print(f"Volatility data shape: {volatilities.shape}")

    # Calculate historical basis
    historical_basis = pricer.calculate_historical_basis(
        tickers=['AAPL', 'MSFT'],
        prices=prices,
        volatilities=volatilities,
        dividends=dividends,
        sofr_rates=sofr_rates,
        cash_rates=cash_rates,
        days=90,
        notional=100000,
        tax_rate=0.30
    )

    if not historical_basis.empty:
        print(f"\nHistorical basis shape: {historical_basis.shape}")
        print(f"\nFirst few rows:")
        print(historical_basis.head())

        # Generate signals
        print("\nGenerating trading signals for AAPL...")
        aapl_basis = historical_basis[historical_basis['ticker'] == 'AAPL']
        aapl_basis_series = aapl_basis.set_index('date')['basis']

        signals = pricer.generate_basis_signals(
            basis_series=aapl_basis_series,
            entry_threshold=0.005,  # 50 bps
            exit_threshold=0.001,  # 10 bps
            lookback_days=20
        )

        # Analyze regimes
        print("\nAnalyzing basis regimes for AAPL...")
        regime_analysis = pricer.analyze_basis_regimes(historical_basis, 'AAPL')

        # Show current trading opportunity
        if len(signals) > 0:
            current_signal = signals['position'].iloc[-1]
            current_basis = signals['basis'].iloc[-1]

            print(f"\nCurrent Trading Opportunity:")
            print(f"  Current basis: {current_basis * 100:.2f}%")
            print(f"  Signal: {current_signal}")

            if current_signal == 1:
                print(f"  → ACTION: Use synthetic (basis is {abs(current_basis) * 100:.2f}% cheap)")
                print(f"  → Expected profit from mean reversion: {signals['expected_return'].iloc[-1] * 100:.2f}%")
            elif current_signal == -1:
                print(f"  → ACTION: Use cash (basis is {current_basis * 100:.2f}% expensive)")
                print(f"  → Expected profit from mean reversion: {abs(signals['expected_return'].iloc[-1]) * 100:.2f}%")
            else:
                print(f"  → ACTION: Wait (basis within normal range)")

    return historical_basis


def test_basis_trading_strategy():
    """Test complete basis trading strategy"""

    print("\nTest: Basis Trading Strategy")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Simulate a trading scenario
    print("Simulating trading scenario...")

    # Create synthetic and cash rates over time
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(42)

    # Synthetic rates (higher and more volatile)
    synthetic_rates = pd.Series(
        0.045 + 0.01 * np.sin(np.arange(200) / 20) + np.random.normal(0, 0.003, 200),
        index=dates
    )

    # Cash rates (lower and less volatile)
    cash_rates = pd.Series(
        0.042 + 0.005 * np.sin(np.arange(200) / 30) + np.random.normal(0, 0.001, 200),
        index=dates
    )

    # Calculate basis
    basis_series = synthetic_rates - cash_rates

    print(f"Basis statistics:")
    print(f"  Mean: {basis_series.mean() * 100:.2f}%")
    print(f"  Std: {basis_series.std() * 100:.2f}%")
    print(f"  Min: {basis_series.min() * 100:.2f}%")
    print(f"  Max: {basis_series.max() * 100:.2f}%")

    # Generate trading signals
    print("\nGenerating trading signals...")
    signals = pricer.generate_basis_signals(
        basis_series=basis_series,
        entry_threshold=0.003,  # 30 bps
        exit_threshold=0.001,  # 10 bps
        lookback_days=50
    )

    # Calculate P&L from following signals
    print("\nCalculating strategy P&L...")

    # Assume we trade $100,000 each time
    notional = 100000

    # Calculate daily returns from basis mean reversion
    # When we're long basis (use synthetic), we profit when basis rises (becomes less negative)
    # When we're short basis (use cash), we profit when basis falls (becomes less positive)
    signals['basis_change'] = basis_series.diff()

    # Calculate P&L
    signals['pnl'] = 0.0

    # Long positions profit when basis increases
    long_positions = signals['position'] == 1
    signals.loc[long_positions, 'pnl'] = signals.loc[long_positions, 'basis_change'] * notional

    # Short positions profit when basis decreases
    short_positions = signals['position'] == -1
    signals.loc[short_positions, 'pnl'] = -signals.loc[short_positions, 'basis_change'] * notional

    # Calculate cumulative P&L
    signals['cumulative_pnl'] = signals['pnl'].cumsum()

    # Calculate statistics
    total_trades = (signals['position'].diff() != 0).sum() - 1
    profitable_trades = (signals.groupby((signals['position'].diff() != 0).cumsum())['pnl'].sum() > 0).sum()

    print(f"\nStrategy Performance:")
    print(f"  Total trades: {total_trades}")
    print(f"  Profitable trades: {profitable_trades}")
    print(f"  Win rate: {profitable_trades / total_trades * 100:.1f}%")
    print(f"  Total P&L: ${signals['cumulative_pnl'].iloc[-1]:.2f}")
    print(f"  Annualized return: {signals['cumulative_pnl'].iloc[-1] / notional * 252 / 200 * 100:.1f}%")

    # Show some trade examples
    print(f"\nExample trades:")
    trade_changes = signals['position'].diff()
    entry_points = trade_changes[trade_changes != 0]

    for i, (date, change) in enumerate(entry_points.head(5).items()):
        basis_value = signals.loc[date, 'basis']
        position = signals.loc[date, 'position']

        if change == 1:
            print(f"  {date.date()}: ENTER LONG at basis {basis_value * 100:.2f}%")
            print(f"     → Synthetic is cheap, use synthetic position")
        elif change == -1:
            print(f"  {date.date()}: ENTER SHORT at basis {basis_value * 100:.2f}%")
            print(f"     → Synthetic is expensive, use cash position")
        elif change == -1 and position == 0:
            print(f"  {date.date()}: EXIT position")

    return signals


def test_integration():
    """Test integration with previous components"""

    print("\nTest: Integration with all components")
    print("=" * 60)

    try:
        from src.data_pipeline import DataPipeline

        # Create data pipeline
        data_pipe = DataPipeline()

        # Fetch data
        print("Fetching market data...")
        prices = data_pipe.fetch_market_data(
            tickers=['AAPL', 'MSFT'],
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
            tickers=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Create synthetic pricer
        pricer = SyntheticPricer()

        # Run total cost analysis first (Component 2.4)
        print("\nRunning total cost analysis...")
        cost_analysis = pricer.batch_cost_analysis(
            tickers=['AAPL', 'MSFT'],
            prices=prices,
            volatilities=volatilities,
            dividends=dividends,
            days=90,
            notional=100000,
            tax_rate=0.30
        )

        if not cost_analysis.empty:
            print(f"\nCost Analysis Results:")
            for _, row in cost_analysis.iterrows():
                print(f"\n{row['ticker']}:")
                print(f"  Synthetic cost: ${row['synthetic_cost']:.2f}")
                print(f"  Cash long cost: ${row['cash_long_cost']:.2f}")
                print(f"  Recommendation: {row['recommendation']}")
                print(f"  Savings: ${row['savings']:.2f}")

                # Calculate basis from the rates
                synthetic_rate = row['financing_rate']
                cash_rate = pricer.sofr_rate  # Assuming cash rate = SOFR

                basis = pricer.calculate_basis(synthetic_rate, cash_rate)
                print(f"  Basis: {basis * 100:.2f}%")

        # Calculate historical basis (Component 2.5)
        print("\nCalculating historical basis...")
        historical_basis = pricer.calculate_historical_basis(
            tickers=['AAPL', 'MSFT'],
            prices=prices,
            volatilities=volatilities,
            dividends=dividends,
            days=90,
            notional=100000,
            tax_rate=0.30
        )

        if not historical_basis.empty:
            # Generate trading signals
            print("\nGenerating trading signals...")
            for ticker in ['AAPL', 'MSFT']:
                ticker_data = historical_basis[historical_basis['ticker'] == ticker]
                if len(ticker_data) > 0:
                    basis_series = ticker_data.set_index('date')['basis']
                    signals = pricer.generate_basis_signals(
                        basis_series=basis_series,
                        entry_threshold=0.005,
                        exit_threshold=0.001,
                        lookback_days=20
                    )

                    # Get current signal
                    if len(signals) > 0:
                        current_signal = signals['position'].iloc[-1]
                        print(f"\n{ticker} current signal: {current_signal}")

        print("\nIntegration test complete!")
        return cost_analysis, historical_basis

    except ImportError as e:
        print(f"Could not import DataPipeline: {e}")
        return None, None