"""
Load and preprocess Lending Club dataset for IFRS 9 ECL Pipeline
"""

import pandas as pd
import numpy as np
import os

# Dataset path
DATASET_PATH = '/Users/zheyuliu/.cache/kagglehub/datasets/wordsforthewise/lending-club/versions/3/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv'

def load_and_preprocess_lending_club(n_samples=None, random_state=42):
    """
    Load Lending Club data and preprocess for IFRS 9 pipeline
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to load (None for all)
    random_state : int
        Random state for sampling
        
    Returns:
    --------
    pd.DataFrame with columns: loan_id, income, total_debt, fico_score, loan_amount, default_flag
    """
    print(f"Loading Lending Club dataset from: {DATASET_PATH}")
    
    # Read data in chunks if needed
    if n_samples:
        # Sample randomly for faster processing
        total_rows = sum(1 for _ in open(DATASET_PATH)) - 1  # Subtract header
        skip = sorted(np.random.RandomState(random_state).choice(
            range(1, total_rows + 1), 
            total_rows - n_samples, 
            replace=False
        ))
        df = pd.read_csv(DATASET_PATH, skiprows=skip)
    else:
        df = pd.read_csv(DATASET_PATH)
    
    print(f"Loaded {len(df):,} rows")
    
    # Map loan_status to default_flag
    # Default = 1 if loan is Charged Off or severely late
    default_mapping = {
        'Fully Paid': 0,
        'Current': 0,
        'In Grace Period': 0,
        'Late (31-120 days)': 1,
        'Default': 1,
        'Charged Off': 1,
        'Does not meet the credit policy. Status:Fully Paid': 0,
        'Does not meet the credit policy. Status:Charged Off': 1
    }
    
    df['default_flag'] = df['loan_status'].map(default_mapping)
    # Fill any unmapped values as 0 (conservative)
    df['default_flag'] = df['default_flag'].fillna(0).astype(int)
    
    # Create loan_id from id column
    df['loan_id'] = 'LOAN_' + df['id'].astype(str)
    
    # Map income (annual_inc)
    df['income'] = df['annual_inc'].fillna(df['annual_inc'].median())
    
    # Calculate total_debt (approximate from dti and annual_inc)
    # DTI = total_debt / annual_inc, so total_debt = dti * annual_inc
    # Use dti if available, otherwise estimate
    if 'dti' in df.columns:
        df['total_debt'] = df['dti'] / 100.0 * df['income']  # dti is percentage
    else:
        # Estimate based on installment and term
        # Rough estimate: monthly debt = installment * 1.5 (to account for other debts)
        df['total_debt'] = df['installment'] * 1.5 * 12  # Annual debt estimate
    
    # Handle missing total_debt
    df['total_debt'] = df['total_debt'].fillna(df['total_debt'].median())
    
    # FICO score: use average of fico_range_low and fico_range_high
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_score'] = ((df['fico_range_low'] + df['fico_range_high']) / 2).fillna(680)
    else:
        # Fallback: estimate from grade
        grade_to_fico = {
            'A': 750, 'B': 700, 'C': 650, 'D': 600, 'E': 550, 'F': 500, 'G': 450
        }
        df['fico_score'] = df['grade'].map(grade_to_fico).fillna(680)
    
    # Loan amount
    df['loan_amount'] = df['loan_amnt'].fillna(df['loan_amnt'].median())
    
    # Select and clean final columns
    result = df[[
        'loan_id',
        'income',
        'total_debt',
        'fico_score',
        'loan_amount',
        'default_flag'
    ]].copy()
    
    # Remove rows with invalid data
    result = result[
        (result['income'] > 0) &
        (result['loan_amount'] > 0) &
        (result['fico_score'] >= 300) &
        (result['fico_score'] <= 850)
    ].copy()
    
    # Round fico_score to integer
    result['fico_score'] = result['fico_score'].astype(int)
    
    print(f"Preprocessed {len(result):,} valid rows")
    print(f"Default rate: {result['default_flag'].mean():.2%}")
    print(f"Average FICO: {result['fico_score'].mean():.0f}")
    print(f"Average income: ${result['income'].mean():,.0f}")
    print(f"Average loan amount: ${result['loan_amount'].mean():,.0f}")
    
    return result

if __name__ == "__main__":
    # Test loading
    df = load_and_preprocess_lending_club(n_samples=10000)
    print("\nSample data:")
    print(df.head())












