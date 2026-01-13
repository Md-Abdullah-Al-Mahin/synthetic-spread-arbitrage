import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic_pricer import SyntheticPricer

def test_calculate_financing_cost():
    """Test the financing cost calculation"""

    print("Testing Component 2.1: calculate_financing_cost()")
    print("=" * 60)

    # Create pricer
    pricer = SyntheticPricer()

    # Test 1: Basic example from documentation
    print("\nTest 1: Basic example (1,000 shares of AAPL at $150)")
    print("-" * 60)

    # Scenario: 1,000 shares of AAPL at $150/share for 90 days
    notional = 1000 * 150  # $150,000
    days = 90

    result = pricer.calculate_financing_cost(
        notional=notional,
        days=days,
        sofr_rate=0.045,  # 4.5%
        spread=0.005  # 50 bps
    )

    print(f"\nExpected calculation:")
    print(f"  (0.045 + 0.005) × $150,000 × (90/360)")
    print(f"  = 0.05 × $150,000 × 0.25")
    print(f"  = $1,875")
    print(f"\nActual result: ${result['total_cost']:,.2f}")

    # Test 2: With defaults
    print("\nTest 2: Using default parameters")
    print("-" * 60)

    result2 = pricer.calculate_financing_cost(
        notional=100000,
        days=30
    )

    # Test 3: Multiple positions
    print("\nTest 3: Multiple positions")
    print("-" * 60)

    notionals = {
        'AAPL': 150000,
        'MSFT': 200000,
        'GOOGL': 100000
    }

    df = pricer.calculate_multiple_positions(notionals, days=90)

    print("\n" + "=" * 60)
    print("Component 2.1 test complete!")

    return result, result2, df