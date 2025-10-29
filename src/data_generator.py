"""
Synthetic panel data generation with temporal structure and drift simulation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta


def make_synthetic_panel(n_policies: int = 2000, months: int = 12, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic monthly panel data for insurance policies.
    
    Simulates realistic relationships:
    - Tenure reduces lapse risk
    - Smoker status increases risk
    - Having an agent reduces risk
    - High premium/coverage ratio increases risk
    - Drift after 2023-07: stronger price sensitivity, reduced agent effect
    
    Includes leakage trap: post_event_flag (excluded from modeling).
    
    Args:
        n_policies: Number of unique policies
        months: Number of months to generate
        seed: Random seed
    
    Returns:
        DataFrame with columns: policy_id, month, age, tenure_m, premium, coverage,
        region, has_agent, is_smoker, dependents, premium_change_rate, region_risk,
        lapse_next_3m, post_event_flag
    """
    np.random.seed(seed)
    
    # Generate base policy characteristics (constant across months)
    policies = pd.DataFrame({
        'policy_id': range(1, n_policies + 1),
        'base_age': np.random.randint(25, 75, n_policies),
        'base_region': np.random.choice(['North', 'South', 'Center', 'East'], n_policies),
        'base_is_smoker': np.random.binomial(1, 0.3, n_policies),
        'base_has_agent': np.random.binomial(1, 0.6, n_policies),
        'base_dependents': np.random.randint(0, 4, n_policies),
        'base_premium': np.random.lognormal(mean=6.0, sigma=0.5, size=n_policies),
        'base_coverage': np.random.lognormal(mean=6.5, sigma=0.4, size=n_policies),
    })
    
    # Region risk coefficients (fixed lookup)
    region_risk_map = {
        'North': 0.15,
        'South': 0.25,
        'Center': 0.10,
        'East': 0.20
    }
    policies['region_risk'] = policies['base_region'].map(region_risk_map)
    
    # Generate monthly panel
    months_list = []
    start_date = datetime(2023, 1, 1)
    
    for m in range(months):
        month_date = start_date + timedelta(days=30 * m)
        month_str = month_date.strftime('%Y-%m')
        months_list.append(month_str)
    
    records = []
    
    # Track previous premiums for change rate calculation
    prev_premiums = {}
    
    for _, pol in policies.iterrows():
        policy_id = pol['policy_id']
        prev_premiums[policy_id] = pol['base_premium']
        
        for month_idx, month_str in enumerate(months_list):
            # Calculate tenure (months since policy start)
            tenure_m = month_idx
            
            # Age increases linearly
            age = pol['base_age'] + month_idx // 12
            
            # Premium can vary slightly month-to-month (with drift)
            premium_noise = np.random.normal(0, 0.02)
            if month_str >= '2023-07':
                # Stronger volatility after drift
                premium_noise = np.random.normal(0, 0.035)
            
            # Get previous premium for change rate calculation
            if month_idx == 0:
                prev_premium = pol['base_premium']
                premium_change_rate = 0.0
            else:
                prev_premium = prev_premiums[policy_id]
                premium_change_rate = (premium - prev_premium) / (prev_premium + 1e-6)
            
            # Calculate new premium
            premium = prev_premiums[policy_id] * (1 + premium_noise)
            
            # Update tracking for next iteration
            prev_premiums[policy_id] = premium
            
            coverage = pol['base_coverage']  # Coverage typically stable
            
            # Check if we're past drift point (2023-07)
            is_post_drift = month_str >= '2023-07'
            
            # Compute base hazard (logistic-style) with STRONGER correlations
            # Non-linear tenure effect: stronger protection early, diminishing returns
            if tenure_m < 6:
                tenure_effect = -0.8  # Strong early protection
            elif tenure_m < 24:
                tenure_effect = -0.4 - 0.02 * (tenure_m - 6)  # Gradual decrease
            else:
                tenure_effect = -0.76 - 0.005 * (tenure_m - 24)  # Very slow decline
            
            # Smoker increases risk (moderately stronger)
            smoker_effect = 0.8 if pol['base_is_smoker'] else 0
            
            # Agent SIGNIFICANTLY reduces risk (stronger effect, but reduced after drift)
            agent_effect = -1.2 if pol['base_has_agent'] else 0
            if is_post_drift and pol['base_has_agent']:
                agent_effect = -0.6  # Still significant but reduced after drift
            
            # Premium/coverage ratio STRONGLY increases risk (much stronger signal)
            premium_coverage_ratio = premium / (coverage + 1e-6)
            price_effect = 0.8 * premium_coverage_ratio * 1000  # Scale for stronger effect
            if is_post_drift:
                price_effect = 1.5 * premium_coverage_ratio * 1000  # MUCH stronger price sensitivity after drift
            
            # Premium change rate effect (increases risk if premium increases)
            premium_change_effect = 2.0 * max(0, premium_change_rate)  # Only positive changes matter
            
            # Dependents reduce risk (more commitment)
            dependents_effect = -0.12 * pol['base_dependents']
            
            # Region risk (fixed coefficient lookup - stronger signal)
            region_risk = pol['region_risk']
            
            # Base hazard with stronger structure
            logit = (-2.5 + tenure_effect + smoker_effect + agent_effect + 
                    price_effect + premium_change_effect + dependents_effect + region_risk)
            
            # Add small noise for realism
            logit += np.random.normal(0, 0.15)
            
            # Convert to probability
            hazard_3m = 1 / (1 + np.exp(-logit))
            hazard_3m = np.clip(hazard_3m, 0.01, 0.95)  # Keep reasonable bounds
            
            # Generate lapse in next 3 months
            lapse_next_3m = np.random.binomial(1, hazard_3m)
            
            # Leakage trap: post_event_flag (correlates with future lapse)
            # This should be excluded from modeling!
            post_event_flag = 1 if (lapse_next_3m == 1 and np.random.random() > 0.3) else 0
            
            records.append({
                'policy_id': policy_id,
                'month': month_str,
                'age': int(age),
                'tenure_m': tenure_m,
                'premium': round(premium, 2),
                'coverage': round(coverage, 2),
                'region': pol['base_region'],
                'has_agent': int(pol['base_has_agent']),
                'is_smoker': int(pol['base_is_smoker']),
                'dependents': int(pol['base_dependents']),
                'premium_change_rate': round(premium_change_rate, 4),
                'region_risk': round(region_risk, 3),
                'lapse_next_3m': lapse_next_3m,
                'post_event_flag': post_event_flag,
            })
    
    df = pd.DataFrame(records)
    return df


def temporal_split(df: pd.DataFrame, train_months: int = 7, val_months: int = 2) -> Dict[str, pd.DataFrame]:
    """
    Split data temporally by month (no shuffle).
    
    Args:
        df: DataFrame with 'month' column
        train_months: Number of months for training
        val_months: Number of months for validation
    
    Returns:
        Dictionary with keys: 'train', 'val', 'test'
    """
    unique_months = sorted(df['month'].unique())
    
    train_end_idx = train_months
    val_end_idx = train_months + val_months
    
    train_months_list = unique_months[:train_end_idx]
    val_months_list = unique_months[train_end_idx:val_end_idx]
    test_months_list = unique_months[val_end_idx:]
    
    train_df = df[df['month'].isin(train_months_list)].copy()
    val_df = df[df['month'].isin(val_months_list)].copy()
    test_df = df[df['month'].isin(test_months_list)].copy()
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def validate_temporal_integrity(split: Dict[str, pd.DataFrame]) -> None:
    """
    Validate temporal integrity of splits:
    1. Train months < Val months < Test months
    2. No forward leakage per policy_id
    
    Args:
        split: Dictionary with 'train', 'val', 'test' DataFrames
    
    Raises:
        AssertionError: If temporal integrity is violated
    """
    train_df = split['train']
    val_df = split['val']
    test_df = split['test']
    
    # Check month ordering
    max_train_month = train_df['month'].max()
    min_val_month = val_df['month'].min()
    min_test_month = test_df['month'].min()
    
    assert max_train_month < min_val_month, \
        f"Temporal violation: max(train_month)={max_train_month} >= min(val_month)={min_val_month}"
    
    assert min_val_month < min_test_month, \
        f"Temporal violation: min(val_month)={min_val_month} >= min(test_month)={min_test_month}"
    
    # Check no forward leakage per policy
    train_policies = set(train_df['policy_id'].unique())
    val_policies = set(val_df['policy_id'].unique())
    test_policies = set(test_df['policy_id'].unique())
    
    # For each policy, check that months don't overlap across splits
    for policy_id in train_policies.intersection(val_policies):
        policy_train_months = set(train_df[train_df['policy_id'] == policy_id]['month'].unique())
        policy_val_months = set(val_df[val_df['policy_id'] == policy_id]['month'].unique())
        overlap = policy_train_months.intersection(policy_val_months)
        assert len(overlap) == 0, \
            f"Forward leakage: policy {policy_id} has overlapping months {overlap} between train and val"
    
    for policy_id in val_policies.intersection(test_policies):
        policy_val_months = set(val_df[val_df['policy_id'] == policy_id]['month'].unique())
        policy_test_months = set(test_df[test_df['policy_id'] == policy_id]['month'].unique())
        overlap = policy_val_months.intersection(policy_test_months)
        assert len(overlap) == 0, \
            f"Forward leakage: policy {policy_id} has overlapping months {overlap} between val and test"
    
    for policy_id in train_policies.intersection(test_policies):
        policy_train_months = set(train_df[train_df['policy_id'] == policy_id]['month'].unique())
        policy_test_months = set(test_df[test_df['policy_id'] == policy_id]['month'].unique())
        overlap = policy_train_months.intersection(policy_test_months)
        assert len(overlap) == 0, \
            f"Forward leakage: policy {policy_id} has overlapping months {overlap} between train and test"
