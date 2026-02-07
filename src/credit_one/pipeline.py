"""
ELT Pipeline for IFRS 9 ECL - Pure Scheduler
Python acts as a pure orchestrator, all business logic is in SQL

Architecture:
- Extract & Load: Python only moves data from CSV to database (raw dump)
- Transform: All data cleaning, feature engineering in SQL (transform_logic.sql)
- Analytics: Python handles statistical modeling only
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os
import sys

# Configuration
REAL_DATA_PATH = "/Users/zheyuliu/.cache/kagglehub/datasets/wordsforthewise/lending-club/versions/3/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
DB_NAME = "bank_risk_elt.db"
SQL_SCRIPT = "transform_logic.sql"

def create_database():
    """Initialize SQLite database"""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"Removed existing {DB_NAME}")
    
    conn = sqlite3.connect(DB_NAME)
    print(f"Created database: {DB_NAME}")
    return conn

def step1_extract_and_load(conn, n_samples=50000):
    """
    Step 1: Extract & Load (Pure Data Movement)
    Python acts as a "mover" - no business logic, just raw dump
    """
    print("\n" + "="*60)
    print("STEP 1: Extract & Load (Raw Data Dump)")
    print("="*60)
    
    if not os.path.exists(REAL_DATA_PATH):
        print(f"❌ Error: Data file not found at {REAL_DATA_PATH}")
        print("Please download the dataset first:")
        print("  python -c \"import kagglehub; kagglehub.dataset_download('wordsforthewise/lending-club')\"")
        sys.exit(1)
    
    print(f"📦 Loading raw data from: {REAL_DATA_PATH}")
    print(f"   (Reading first {n_samples:,} rows for demonstration)")
    
    # Read CSV in chunks (data is large)
    # We read the first chunk only for demonstration
    chunk_size = n_samples
    df_iter = pd.read_csv(REAL_DATA_PATH, chunksize=chunk_size, low_memory=False)
    
    # Get first chunk
    df_raw = next(df_iter)
    
    print(f"   Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
    
    # Raw dump: No cleaning, no transformation, just load as-is
    # This simulates data landing from external systems
    df_raw.to_sql('raw_landing_zone', conn, if_exists='replace', index=False)
    conn.commit()
    
    print(f"✅ Raw data loaded into 'raw_landing_zone' table")
    print(f"   Columns: {', '.join(df_raw.columns[:10].tolist())}... ({len(df_raw.columns)} total)")
    
    return conn

def step2_transform_sql(conn):
    """
    Step 2: Transform (Pure SQL Processing)
    All data cleaning and feature engineering happens in SQL
    """
    print("\n" + "="*60)
    print("STEP 2: Transform (SQL Processing)")
    print("="*60)
    
    if not os.path.exists(SQL_SCRIPT):
        print(f"❌ Error: SQL script not found: {SQL_SCRIPT}")
        sys.exit(1)
    
    print(f"🔧 Executing SQL transformation script: {SQL_SCRIPT}")
    
    with open(SQL_SCRIPT, 'r') as sql_file:
        sql_script = sql_file.read()
    
    # Split by semicolon and execute statements one by one
    # SQLite doesn't support multiple statements in executescript well for views
    statements = [s.strip() + ';' for s in sql_script.split(';') if s.strip() and not s.strip().startswith('--')]
    
    cursor = conn.cursor()
    executed = 0
    
    for statement in statements:
        if statement.strip() and len(statement.strip()) > 1:
            try:
                cursor.execute(statement)
                executed += 1
            except sqlite3.OperationalError as e:
                # Some statements might fail (like DROP IF EXISTS on non-existent objects)
                # This is okay for idempotent scripts
                if 'no such table' not in str(e).lower() and 'no such view' not in str(e).lower():
                    print(f"   ⚠️  Warning: {str(e)[:100]}")
                continue
    
    conn.commit()
    print(f"✅ Executed {executed} SQL statements")
    
    # Verify tables/views were created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('clean_loans', 'raw_landing_zone')")
    tables = cursor.fetchall()
    print(f"   Created tables: {[t[0] for t in tables]}")
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='model_features'")
    views = cursor.fetchall()
    print(f"   Created views: {[v[0] for v in views]}")
    
    return conn

def step3_modeling(conn):
    """
    Step 3: Statistical Modeling (Python takes over for math)
    Pull engineered features from SQL view, train model, write predictions back
    """
    print("\n" + "="*60)
    print("STEP 3: Statistical Modeling (Python)")
    print("="*60)
    
    # Pull engineered features from SQL view
    print("📊 Fetching engineered features from SQL view 'model_features'...")
    query = """
    SELECT 
        loan_id,
        dti_ratio,
        CASE credit_bucket
            WHEN 'Excellent' THEN 4
            WHEN 'Good' THEN 3
            WHEN 'Fair' THEN 2
            WHEN 'Poor' THEN 1
            ELSE 0
        END AS credit_bucket_encoded,
        CASE loan_size_category
            WHEN 'Large' THEN 2
            WHEN 'Medium' THEN 1
            ELSE 0
        END AS loan_size_encoded,
        CASE income_category
            WHEN 'High' THEN 2
            WHEN 'Medium' THEN 1
            ELSE 0
        END AS income_category_encoded,
        default_flag
    FROM model_features
    WHERE fico_score IS NOT NULL
    """
    
    df = pd.read_sql(query, conn)
    print(f"✅ Loaded {len(df):,} rows from SQL view")
    
    if len(df) == 0:
        print("❌ Error: No data in model_features view")
        return conn, None
    
    # Prepare features and target
    X = df[['dti_ratio', 'credit_bucket_encoded', 'loan_size_encoded', 'income_category_encoded']]
    y = df['default_flag']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Logistic Regression
    print("🤖 Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Model performance
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✅ Model trained")
    print(f"   Test AUC: {auc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default'], zero_division=0))
    
    # Calculate PD for all loans
    all_predictions = model.predict_proba(X)[:, 1]
    
    # Write predictions back to SQL
    predictions_df = pd.DataFrame({
        'loan_id': df['loan_id'],
        'pd_12m': all_predictions,
        'pd_lifetime': all_predictions * 1.2,  # Simplified: lifetime = 12m * 1.2
        'model_version': 'v1.0'
    })
    
    predictions_df.to_sql('loan_predictions', conn, if_exists='replace', index=False)
    conn.commit()
    
    print(f"✅ Wrote {len(predictions_df):,} predictions to 'loan_predictions' table")
    print(f"   Average PD: {predictions_df['pd_12m'].mean():.4f}")
    
    return conn, model

def step4_ifrs9_staging_sql(conn):
    """
    Step 4: IFRS 9 Staging (Back to SQL for business logic)
    All staging rules implemented in SQL
    """
    print("\n" + "="*60)
    print("STEP 4: IFRS 9 Staging (SQL Business Logic)")
    print("="*60)
    
    # Create staging table using SQL
    staging_query = """
    CREATE TABLE IF NOT EXISTS loan_staging AS
    SELECT 
        mf.loan_id,
        mf.loan_amount,
        mf.default_flag,
        mf.grade,
        mf.fico_score,
        lp.pd_12m,
        lp.pd_lifetime,
        
        -- IFRS 9 Stage Assignment (SQL CASE WHEN logic)
        CASE 
            WHEN mf.default_flag = 1 THEN 3  -- Stage 3: Defaulted
            WHEN lp.pd_12m >= 0.02 THEN 2     -- Stage 2: Significant increase in credit risk
            ELSE 1                             -- Stage 1: Performing
        END AS stage,
        
        -- ECL Calculation: PD × LGD × EAD
        -- LGD = 0.45 (45% loss given default)
        CASE 
            WHEN mf.default_flag = 1 THEN mf.loan_amount * 0.45  -- Stage 3: Full ECL
            WHEN lp.pd_12m >= 0.02 THEN mf.loan_amount * lp.pd_lifetime * 0.45  -- Stage 2: Lifetime ECL
            ELSE mf.loan_amount * lp.pd_12m * 0.45 * 0.5  -- Stage 1: 12-month ECL (simplified)
        END AS ecl
        
    FROM model_features mf
    INNER JOIN loan_predictions lp ON mf.loan_id = lp.loan_id
    """
    
    conn.execute("DROP TABLE IF EXISTS loan_staging")
    conn.execute(staging_query)
    conn.commit()
    
    print("✅ Created 'loan_staging' table with IFRS 9 staging logic")
    
    # Show stage distribution
    stage_dist = pd.read_sql("""
        SELECT 
            stage,
            CASE stage
                WHEN 1 THEN 'Stage 1: Performing'
                WHEN 2 THEN 'Stage 2: Significant Increase in Credit Risk'
                WHEN 3 THEN 'Stage 3: Defaulted'
            END AS stage_description,
            COUNT(*) AS loan_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM loan_staging), 2) AS pct,
            ROUND(SUM(ecl), 2) AS total_ecl,
            ROUND(AVG(pd_12m), 4) AS avg_pd
        FROM loan_staging
        GROUP BY stage
        ORDER BY stage
    """, conn)
    
    print("\n  Stage Distribution:")
    print(stage_dist.to_string(index=False))
    
    return conn

def step5_reporting(conn):
    """
    Step 5: Portfolio Risk Reporting (SQL Aggregations)
    """
    print("\n" + "="*60)
    print("STEP 5: Portfolio Risk Reporting (SQL)")
    print("="*60)
    
    # Portfolio summary
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
            ROUND(AVG(pd_12m), 4) AS avg_pd
        FROM loan_staging
        GROUP BY stage
        ORDER BY stage
    """, conn)
    
    print("\n  Portfolio ECL Summary by Stage:")
    print(portfolio_summary.to_string(index=False))
    
    # Overall metrics
    overall = pd.read_sql("""
        SELECT 
            COUNT(*) AS total_loans,
            ROUND(SUM(loan_amount), 2) AS total_exposure,
            ROUND(SUM(ecl), 2) AS total_ecl,
            ROUND(SUM(ecl) / SUM(loan_amount) * 100, 2) AS portfolio_ecl_rate_pct,
            ROUND(AVG(pd_12m), 4) AS portfolio_avg_pd
        FROM loan_staging
    """, conn)
    
    print("\n  Overall Portfolio Metrics:")
    print(overall.to_string(index=False))
    
    return conn

def main():
    """Main ELT Pipeline Execution"""
    print("\n" + "="*60)
    print("IFRS 9 ECL Pipeline - ELT Architecture")
    print("Extract & Load (Python) | Transform (SQL) | Analytics (Python)")
    print("="*60)
    
    # Parse command line arguments
    n_samples = 50000
    if '--samples' in sys.argv:
        idx = sys.argv.index('--samples')
        if idx + 1 < len(sys.argv):
            n_samples = int(sys.argv[idx + 1])
    
    try:
        # Initialize database
        conn = create_database()
        
        # Execute ELT pipeline
        conn = step1_extract_and_load(conn, n_samples=n_samples)
        conn = step2_transform_sql(conn)
        conn, model = step3_modeling(conn)
        conn = step4_ifrs9_staging_sql(conn)
        conn = step5_reporting(conn)
        
        print("\n" + "="*60)
        print("✅ ELT Pipeline completed successfully!")
        print(f"✅ Database saved: {DB_NAME}")
        print("="*60)
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
IFRS 9 ECL Pipeline - ELT Architecture

Usage:
  python pipeline.py [--samples N]

Options:
  --samples N    Number of samples to load from dataset (default: 50000)
  -h, --help     Show this help message

Architecture:
  - Extract & Load: Python moves raw CSV to database (no transformation)
  - Transform: All cleaning/feature engineering in SQL (transform_logic.sql)
  - Analytics: Python handles statistical modeling only
        """)
        sys.exit(0)
    main()












