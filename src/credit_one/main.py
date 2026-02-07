"""
IFRS 9 ECL Pipeline - Database-Centric Risk Pipeline
Demonstrates SQL (Data Engineering) + Python (Modeling) integration

This pipeline simulates a real banking environment where:
- Data engineering (ETL, feature engineering) is done in SQL
- Statistical modeling is done in Python
- Results are stored and processed in SQL for auditability
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os
import sys

# Database configuration
DB_NAME = 'credit_risk.db'

# Try to import lending club data loader
try:
    from load_lending_club_data import load_and_preprocess_lending_club
    LENDING_CLUB_AVAILABLE = True
except ImportError:
    LENDING_CLUB_AVAILABLE = False
    print("Note: Lending Club data loader not available. Using synthetic data only.")

def create_database():
    """Initialize SQLite database and remove existing database if present"""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"Removed existing {DB_NAME}")
    
    conn = sqlite3.connect(DB_NAME)
    print(f"Created database: {DB_NAME}")
    return conn

def step1_raw_data_injection(conn, use_real_data=False, n_samples=10000):
    """
    Step 1: Load raw loan data and inject into SQL table
    This simulates data ingestion from external systems
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    use_real_data : bool
        If True, use Lending Club real data; if False, use synthetic data
    n_samples : int
        Number of samples to use (for synthetic data or sampling real data)
    """
    print("\n" + "="*60)
    print("STEP 1: Raw Data Injection")
    print("="*60)
    
    if use_real_data and LENDING_CLUB_AVAILABLE:
        print("Using REAL Lending Club data")
        # Load real data
        raw_data = load_and_preprocess_lending_club(n_samples=n_samples, random_state=42)
        default_rate = raw_data['default_flag'].mean()
        avg_fico = raw_data['fico_score'].mean()
    else:
        print("Using SYNTHETIC data")
        # Generate synthetic loan data
        np.random.seed(42)
        
        # Generate realistic loan data
        income = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)  # Annual income
        total_debt = np.random.lognormal(mean=9.0, sigma=0.6, size=n_samples)  # Total debt
        fico_score = np.random.normal(loc=680, scale=80, size=n_samples)
        fico_score = np.clip(fico_score, 300, 850)  # FICO range
        loan_amount = np.random.lognormal(mean=10.0, sigma=0.7, size=n_samples)
        
        # Create default flag with realistic probability based on features
        dti_ratio = total_debt / income
        default_prob = 1 / (1 + np.exp(-(-3 + 2*dti_ratio - 0.01*fico_score + 0.0001*loan_amount)))
        default_flag = (np.random.random(n_samples) < default_prob).astype(int)
        
        # Create DataFrame
        raw_data = pd.DataFrame({
            'loan_id': [f'LOAN_{i:06d}' for i in range(1, n_samples + 1)],
            'income': income,
            'total_debt': total_debt,
            'fico_score': fico_score.astype(int),
            'loan_amount': loan_amount,
            'default_flag': default_flag
        })
        default_rate = default_flag.mean()
        avg_fico = fico_score.mean()
    
    # Load into SQL table
    raw_data.to_sql('raw_loans', conn, if_exists='replace', index=False)
    
    print(f"✓ Injected {len(raw_data):,} rows into 'raw_loans' table")
    print(f"  Default rate: {default_rate:.2%}")
    print(f"  Average FICO: {avg_fico:.0f}")
    
    return conn

def step2_sql_feature_engineering(conn):
    """
    Step 2: SQL Feature Engineering (The "DE" part)
    All feature engineering is done in SQL to demonstrate data engineering skills
    """
    print("\n" + "="*60)
    print("STEP 2: SQL Feature Engineering")
    print("="*60)
    
    # Create model_features table using SQL
    # This demonstrates SQL CASE WHEN for categorical binning
    create_features_query = """
    CREATE TABLE model_features AS
    SELECT 
        loan_id,
        income,
        total_debt,
        fico_score,
        loan_amount,
        default_flag,
        -- Feature Engineering: Calculate DTI Ratio
        ROUND(total_debt / NULLIF(income, 0), 4) AS dti_ratio,
        -- Feature Engineering: FICO Score Bucketing (WoE-style binning)
        CASE 
            WHEN fico_score < 580 THEN 'Poor'
            WHEN fico_score < 670 THEN 'Fair'
            WHEN fico_score < 740 THEN 'Good'
            ELSE 'Excellent'
        END AS fico_category,
        -- Additional features
        ROUND(loan_amount / NULLIF(income, 0), 4) AS loan_to_income_ratio,
        CASE 
            WHEN loan_amount > 50000 THEN 'Large'
            WHEN loan_amount > 20000 THEN 'Medium'
            ELSE 'Small'
        END AS loan_size_category
    FROM raw_loans
    """
    
    conn.execute(create_features_query)
    conn.commit()
    
    # Verify the table
    result = conn.execute("SELECT COUNT(*) FROM model_features").fetchone()
    print(f"✓ Created 'model_features' table with {result[0]:,} rows")
    
    # Show sample of engineered features
    sample = pd.read_sql("""
        SELECT 
            loan_id,
            dti_ratio,
            fico_category,
            loan_to_income_ratio,
            default_flag
        FROM model_features
        LIMIT 5
    """, conn)
    print("\n  Sample engineered features:")
    print(sample.to_string(index=False))
    
    return conn

def step3_python_modeling(conn):
    """
    Step 3: Python Modeling
    Pull data from SQL, train model, calculate PD
    """
    print("\n" + "="*60)
    print("STEP 3: Python Modeling")
    print("="*60)
    
    # Pull data from SQL
    query = """
    SELECT 
        dti_ratio,
        CASE fico_category
            WHEN 'Poor' THEN 0
            WHEN 'Fair' THEN 1
            WHEN 'Good' THEN 2
            ELSE 3
        END AS fico_encoded,
        loan_to_income_ratio,
        CASE loan_size_category
            WHEN 'Small' THEN 0
            WHEN 'Medium' THEN 1
            ELSE 2
        END AS loan_size_encoded,
        default_flag
    FROM model_features
    """
    
    df = pd.read_sql(query, conn)
    print(f"✓ Loaded {len(df):,} rows from SQL for modeling")
    
    # Prepare features and target
    X = df[['dti_ratio', 'fico_encoded', 'loan_to_income_ratio', 'loan_size_encoded']]
    y = df['default_flag']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Model performance
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✓ Model trained successfully")
    print(f"  Test AUC: {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    # Calculate PD for all loans
    all_predictions = model.predict_proba(df[['dti_ratio', 'fico_encoded', 
                                                'loan_to_income_ratio', 'loan_size_encoded']])[:, 1]
    
    # Get loan_ids for mapping
    loan_ids = pd.read_sql("SELECT loan_id FROM model_features", conn)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'loan_id': loan_ids['loan_id'],
        'pd': all_predictions
    })
    
    # Write predictions back to SQL
    predictions_df.to_sql('loan_predictions', conn, if_exists='replace', index=False)
    conn.commit()
    
    print(f"\n✓ Wrote {len(predictions_df):,} PD predictions to 'loan_predictions' table")
    print(f"  Average PD: {predictions_df['pd'].mean():.4f}")
    print(f"  PD range: [{predictions_df['pd'].min():.4f}, {predictions_df['pd'].max():.4f}]")
    
    return conn, model

def step4_sql_post_processing_staging(conn):
    """
    Step 4: SQL Post-Processing (Staging)
    Calculate ECL and assign IFRS 9 Stages using SQL logic
    This is the CRITICAL part that demonstrates SQL CASE WHEN for business rules
    """
    print("\n" + "="*60)
    print("STEP 4: SQL Post-Processing (IFRS 9 Staging)")
    print("="*60)
    
    # Create staging table with ECL calculation
    # CRITICAL: All staging logic is done in SQL, not Pandas
    staging_query = """
    CREATE TABLE loan_staging AS
    SELECT 
        mf.loan_id,
        mf.loan_amount,
        mf.default_flag,
        lp.pd,
        -- IFRS 9 Stage Assignment (SQL CASE WHEN logic)
        CASE 
            WHEN mf.default_flag = 1 THEN 3  -- Stage 3: Defaulted
            WHEN lp.pd >= 0.02 THEN 2        -- Stage 2: Significant increase in credit risk
            ELSE 1                            -- Stage 1: Performing
        END AS stage,
        -- ECL Calculation (simplified: PD * LGD * EAD)
        -- Assumptions: LGD = 0.45 (45% loss given default), EAD = loan_amount
        CASE 
            WHEN mf.default_flag = 1 THEN mf.loan_amount * 0.45  -- Stage 3: Full ECL
            WHEN lp.pd >= 0.02 THEN mf.loan_amount * lp.pd * 0.45  -- Stage 2: Lifetime ECL
            ELSE mf.loan_amount * lp.pd * 0.45 * 0.5  -- Stage 1: 12-month ECL (simplified as 50% of lifetime)
        END AS ecl
    FROM model_features mf
    INNER JOIN loan_predictions lp ON mf.loan_id = lp.loan_id
    """
    
    conn.execute(staging_query)
    conn.commit()
    
    # Verify staging results
    result = conn.execute("SELECT COUNT(*) FROM loan_staging").fetchone()
    print(f"✓ Created 'loan_staging' table with {result[0]:,} rows")
    
    # Show stage distribution
    stage_dist = pd.read_sql("""
        SELECT 
            stage,
            COUNT(*) AS loan_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM loan_staging), 2) AS pct,
            ROUND(SUM(ecl), 2) AS total_ecl,
            ROUND(AVG(pd), 4) AS avg_pd
        FROM loan_staging
        GROUP BY stage
        ORDER BY stage
    """, conn)
    
    print("\n  Stage Distribution:")
    print(stage_dist.to_string(index=False))
    
    # Show sample staging results
    sample = pd.read_sql("""
        SELECT 
            loan_id,
            loan_amount,
            pd,
            stage,
            ROUND(ecl, 2) AS ecl
        FROM loan_staging
        WHERE stage IN (1, 2, 3)
        ORDER BY stage, pd DESC
        LIMIT 10
    """, conn)
    
    print("\n  Sample Staging Results:")
    print(sample.to_string(index=False))
    
    return conn

def step5_reporting(conn):
    """
    Step 5: Reporting
    Use SQL queries to generate portfolio risk reports
    """
    print("\n" + "="*60)
    print("STEP 5: Portfolio Risk Reporting")
    print("="*60)
    
    # Portfolio-level ECL summary
    portfolio_summary = pd.read_sql("""
        SELECT 
            stage,
            CASE stage
                WHEN 1 THEN 'Stage 1: Performing'
                WHEN 2 THEN 'Stage 2: Significant Increase in Credit Risk'
                WHEN 3 THEN 'Stage 3: Defaulted'
            END AS stage_description,
            COUNT(*) AS loan_count,
            ROUND(SUM(loan_amount), 2) AS total_exposure,
            ROUND(SUM(ecl), 2) AS total_ecl,
            ROUND(SUM(ecl) / SUM(loan_amount) * 100, 2) AS ecl_rate_pct,
            ROUND(AVG(pd), 4) AS avg_pd
        FROM loan_staging
        GROUP BY stage
        ORDER BY stage
    """, conn)
    
    print("\n  Portfolio ECL Summary by Stage:")
    print(portfolio_summary.to_string(index=False))
    
    # Overall portfolio metrics
    overall = pd.read_sql("""
        SELECT 
            COUNT(*) AS total_loans,
            ROUND(SUM(loan_amount), 2) AS total_exposure,
            ROUND(SUM(ecl), 2) AS total_ecl,
            ROUND(SUM(ecl) / SUM(loan_amount) * 100, 2) AS portfolio_ecl_rate_pct,
            ROUND(AVG(pd), 4) AS portfolio_avg_pd,
            SUM(CASE WHEN stage = 3 THEN 1 ELSE 0 END) AS defaulted_loans
        FROM loan_staging
    """, conn)
    
    print("\n  Overall Portfolio Metrics:")
    print(overall.to_string(index=False))
    
    # Top 10 highest ECL loans
    top_ecl = pd.read_sql("""
        SELECT 
            loan_id,
            loan_amount,
            pd,
            stage,
            ROUND(ecl, 2) AS ecl
        FROM loan_staging
        ORDER BY ecl DESC
        LIMIT 10
    """, conn)
    
    print("\n  Top 10 Highest ECL Loans:")
    print(top_ecl.to_string(index=False))
    
    return conn

def main():
    """Main pipeline execution"""
    print("\n" + "="*60)
    print("IFRS 9 ECL Pipeline - Database-Centric Risk Pipeline")
    print("SQL (Data Engineering) + Python (Modeling) Integration")
    print("="*60)
    
    # Check command line arguments
    use_real_data = '--real-data' in sys.argv or '--lending-club' in sys.argv
    n_samples = 10000
    if '--samples' in sys.argv:
        idx = sys.argv.index('--samples')
        if idx + 1 < len(sys.argv):
            n_samples = int(sys.argv[idx + 1])
    
    try:
        # Initialize database
        conn = create_database()
        
        # Execute pipeline steps
        conn = step1_raw_data_injection(conn, use_real_data=use_real_data, n_samples=n_samples)
        conn = step2_sql_feature_engineering(conn)
        conn, model = step3_python_modeling(conn)
        conn = step4_sql_post_processing_staging(conn)
        conn = step5_reporting(conn)
        
        print("\n" + "="*60)
        print("✓ Pipeline completed successfully!")
        print(f"✓ Database saved: {DB_NAME}")
        print("="*60)
        
        # Close connection
        conn.close()
        
    except Exception as e:
        print(f"\n✗ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
IFRS 9 ECL Pipeline - Usage

Options:
  --real-data, --lending-club    Use real Lending Club data instead of synthetic
  --samples N                    Number of samples to use (default: 10000)
  -h, --help                     Show this help message

Examples:
  python main.py                          # Use synthetic data (10k samples)
  python main.py --real-data               # Use real Lending Club data (10k samples)
  python main.py --real-data --samples 50000  # Use real data with 50k samples
        """)
        sys.exit(0)
    main()

