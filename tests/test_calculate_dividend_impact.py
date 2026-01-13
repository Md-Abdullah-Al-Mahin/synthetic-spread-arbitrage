import sys
import os
import pandas as pd
import numpy as np
from typing import Any


# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic_pricer import SyntheticPricer

def create_sample_data():
    """Create sample price and dividend data for testing"""

    # Sample prices
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    prices = pd.DataFrame({
        'AAPL': [185.0, 186.5, 187.2, 188.0, 189.5],
        'MSFT': [410.0, 412.5, 411.0, 413.8, 415.0],
        'GOOGL': [145.0, 146.2, 147.0, 145.8, 148.0],
        'TSLA': [240.0, 242.5, 241.0, 239.5, 238.0]
    }, index=dates)

    # Create dividend data with same index
    # All arrays must be the same length (5 in this case)
    dividends = pd.DataFrame({
        'AAPL': [0.24, 0.24, 0.24, 0.24, np.nan],  # 4 payments + 1 NaN
        'MSFT': [0.68, 0.68, 0.68, 0.75, np.nan],  # 4 payments + 1 NaN
        'GOOGL': [0.20, np.nan, np.nan, np.nan, np.nan],  # 1 payment + 4 NaN
        'TSLA': [np.nan, np.nan, np.nan, np.nan, np.nan]  # All NaN
    }, index=dates)

    return prices, dividends


def test_dividend_impact():
    """Test dividend impact calculation"""

    print("Testing Component 2.3: calculate_dividend_impact()")
    print("=" * 60)

    pricer = SyntheticPricer()

    # Create sample data
    print("\nCreating sample data...")
    prices, dividends = create_sample_data()

    print(f"Price data shape: {prices.shape}")
    print(f"Dividend data shape: {dividends.shape}")

    # Test 1: Basic dividend impact calculation
    print("\nTest 1: Basic dividend impact")
    print("-" * 60)

    result = pricer.calculate_dividend_impact(
        prices=prices,
        dividends=dividends,
        tax_rate=0.30,
        position_days=90
    )

    print(f"\nResult DataFrame shape: {result.shape}")

    # Test 2: MSFT example from documentation
    print("\nTest 2: MSFT example from documentation")
    print("-" * 60)

    # Example from documentation:
    # MSFT: Annual Dividend Yield: 2.5% (0.025), Tax Rate: 30% (0.30)
    # Cash: 2.5% × (1 - 0.30) = 1.75%
    # Synthetic: 2.5% (gross)
    # Advantage = 2.5% - 1.75% = 0.75%

    # Create mock data for MSFT
    msft_prices = pd.DataFrame({
        'MSFT': [400.0]
    })

    # Annual dividend = 2.5% of $400 = $10
    msft_dividends = pd.DataFrame({
        'MSFT': [10.0]  # $10 annual dividend
    }, index=[pd.Timestamp('2024-12-31')])

    # Calculate with 30% tax rate
    msft_result = pricer.calculate_dividend_impact(
        prices=msft_prices,
        dividends=msft_dividends,
        tax_rate=0.30,
        position_days=90
    )

    if not msft_result.empty:
        advantage = msft_result.loc['MSFT', 'advantage'] * 100

        print(f"\nMSFT Calculation:")
        print(f"  Expected advantage: 0.75%")
        print(f"  Calculated advantage: {advantage:.2f}%")

        if abs(advantage - 0.75) < 0.1:
            print("Advantage calculation is correct")
        else:
            print("Advantage calculation differs")

        # Check dollar amounts
        annual_advantage = msft_result.loc['MSFT', 'annual_advantage']
        expected_annual = 0.0075 * 100000  # 0.75% of $100k
        print(f"\n  Expected annual advantage on $100k: ${expected_annual:.2f}")
        print(f"  Calculated annual advantage: ${annual_advantage:.2f}")

    # Test 3: Different tax rates
    print("\nTest 3: Different tax rates")
    print("-" * 60)

    test_data = pd.DataFrame({
        'ticker': ['TEST'],
        'current_price': [100.0],
        'annual_dividend': [5.0],  # 5% yield
    })

    for tax_rate in [0.15, 0.20, 0.30, 0.40]:
        # Manual calculation
        dividend_yield = 5.0 / 100.0  # 5%
        cash_yield = dividend_yield * (1 - tax_rate)
        synthetic_yield = dividend_yield * 1.0
        advantage = synthetic_yield - cash_yield

        print(f"\nTax rate {tax_rate * 100}%:")
        print(f"  Cash yield: {cash_yield * 100:.2f}%")
        print(f"  Synthetic yield: {synthetic_yield * 100:.2f}%")
        print(f"  Advantage: {advantage * 100:.2f}%")
        print(f"  Advantage as % of dividend: {(advantage / dividend_yield) * 100:.1f}%")

    # Test 4: Net financing cost with dividend impact
    print("\nTest 4: Net financing cost calculation")
    print("-" * 60)

    net_cost = pricer.calculate_net_financing_cost(
        notional=100000,
        days=90,
        volatility=0.25,  # 25% vol
        dividend_yield=0.025,  # 2.5% yield
        tax_rate=0.30
    )

    print("\nNet cost breakdown:")
    for key, value in net_cost.items():
        if 'rate' in key or 'yield' in key:
            print(f"  {key}: {value * 100:.2f}%")
        elif 'cost' in key or 'advantage' in key:
            print(f"  {key}: ${value:.2f}")

    # Test 5: Integration with real data
    print("\nTest 5: Integration with Data Pipeline")
    print("-" * 60)

    try:
        from src.data_pipeline import DataPipeline

        # Create data pipeline
        data_pipe = DataPipeline()

        # Fetch data for dividend-paying stocks
        print("Fetching data for AAPL and MSFT...")
        prices_real = data_pipe.fetch_market_data(
            tickers=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2024-06-30'
        )

        # Get dividend data
        print("Fetching dividend data...")
        dividends_real = data_pipe.get_dividend_data(
            tickers=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2024-06-30'
        )

        if not dividends_real.empty:
            print("\nCalculating dividend impact...")
            impact = pricer.calculate_dividend_impact(
                prices=prices_real,
                dividends=dividends_real,
                tax_rate=0.30,
                position_days=90
            )

            # Calculate volatility for net cost
            returns = data_pipe.calculate_returns(prices_real)
            volatility_data = data_pipe.calculate_realized_volatility(returns, window=30)

            print("\nNet financing cost for each:")
            for ticker in ['AAPL', 'MSFT']:
                if ticker in impact.index and ticker in volatility_data.columns:
                    vol = volatility_data[ticker].iloc[-1]
                    div_yield = impact.loc[ticker, 'dividend_yield']

                    net = pricer.calculate_net_financing_cost(
                        notional=100000,
                        days=90,
                        volatility=vol,
                        dividend_yield=div_yield,
                        tax_rate=0.30
                    )

        else:
            print("No dividend data available")

    except ImportError as e:
        print(f"Could not import DataPipeline: {e}")

    print("\n" + "=" * 60)
    print("Component 2.3 tests complete!")

    return result, net_cost