import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from statistical_models import StatisticalModels


def create_test_savings_data():
    """Create sample savings data"""
    np.random.seed(42)

    # Case 1: Clearly positive savings
    positive_savings = np.random.normal(0.0015, 0.0005, 30)  # ~15 bps mean

    # Case 2: No real savings (around zero)
    no_savings = np.random.normal(0.0001, 0.0005, 30)  # ~1 bps mean

    # Case 3: Negative savings (strategy loses money)
    negative_savings = np.random.normal(-0.0010, 0.0005, 30)  # -10 bps mean

    return {
        'positive': positive_savings,
        'no_savings': no_savings,
        'negative': negative_savings
    }


def test_basic_hypothesis():
    """Test basic hypothesis testing"""
    print("Test: Basic Hypothesis Testing")
    print("=" * 50)

    test_data = create_test_savings_data()
    model = StatisticalModels()

    # Test 1: Positive savings
    print("\n1. Testing Positive Savings:")
    print(f"   Mean: {np.mean(test_data['positive']):.6f} ({np.mean(test_data['positive']) * 10000:.1f} bps)")
    results1 = model.hypothesis_test_cost_savings(test_data['positive'])
    model.print_hypothesis_test_summary(results1)

    # Test 2: No savings
    print("\n2. Testing No Significant Savings:")
    print(f"   Mean: {np.mean(test_data['no_savings']):.6f} ({np.mean(test_data['no_savings']) * 10000:.1f} bps)")
    results2 = model.hypothesis_test_cost_savings(test_data['no_savings'])
    model.print_hypothesis_test_summary(results2)

    # Test 3: Negative savings
    print("\n3. Testing Negative Savings:")
    print(f"   Mean: {np.mean(test_data['negative']):.6f} ({np.mean(test_data['negative']) * 10000:.1f} bps)")
    results3 = model.hypothesis_test_cost_savings(test_data['negative'])
    model.print_hypothesis_test_summary(results3)

    return results1, results2, results3


def test_realistic_scenario():
    """Test with realistic trading data"""
    print("\n\nRealistic Trading Scenario")
    print("=" * 50)

    np.random.seed(123)

    # Simulate 50 trades
    n_trades = 50

    # Base savings: 12.3 bps as in README example
    base_savings = 0.00123  # 12.3 bps

    # Add noise
    savings = base_savings + np.random.normal(0, 0.00078, n_trades)  # 7.8 bps std

    print(f"Simulating {n_trades} trades")
    print(f"Expected mean savings: {base_savings:.6f} ({base_savings * 10000:.1f} bps)")
    print(f"Sample mean: {np.mean(savings):.6f} ({np.mean(savings) * 10000:.1f} bps)")
    print(f"Sample std: {np.std(savings):.6f} ({np.std(savings) * 10000:.1f} bps)")

    model = StatisticalModels()
    results = model.hypothesis_test_cost_savings(savings)

    print("\nStatistical Test:")
    model.print_hypothesis_test_summary(results)

    # Compare with README example
    print(f"\nComparison with README example:")
    print(f"  README: t = 10.0, p < 0.0001")
    print(f"  Our test: t = {results['t_stat']:.1f}, p = {results['p_value']:.4f}")

    return results


def test_edge_cases():
    """Test edge cases"""
    print("\n\nEdge Cases")
    print("=" * 50)

    model = StatisticalModels()

    # Test 1: Very few observations
    print("\n1. Few observations (n=3):")
    few_data = [0.001, 0.002, 0.0015]
    results1 = model.hypothesis_test_cost_savings(few_data)
    model.print_hypothesis_test_summary(results1)

    # Test 2: Constant data
    print("\n2. Constant data (all same):")
    constant_data = [0.001] * 10
    results2 = model.hypothesis_test_cost_savings(constant_data)
    model.print_hypothesis_test_summary(results2)

    # Test 3: Very large savings
    print("\n3. Very large savings:")
    large_savings = np.random.normal(0.005, 0.001, 20)  # ~50 bps
    results3 = model.hypothesis_test_cost_savings(large_savings)
    model.print_hypothesis_test_summary(results3)