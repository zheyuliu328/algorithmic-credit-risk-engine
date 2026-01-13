-- IFRS 9 ECL Pipeline - Database Schema
-- This file documents the table structures used in the pipeline

-- ============================================
-- Table: raw_loans
-- Purpose: Raw loan data ingestion
-- Source: Synthetic data generation (simulates external data source)
-- ============================================
CREATE TABLE raw_loans (
    loan_id TEXT PRIMARY KEY,
    income REAL,
    total_debt REAL,
    fico_score INTEGER,
    loan_amount REAL,
    default_flag INTEGER
);

-- ============================================
-- Table: model_features
-- Purpose: Feature-engineered data for modeling
-- Source: SQL transformation from raw_loans
-- ============================================
CREATE TABLE model_features (
    loan_id TEXT PRIMARY KEY,
    income REAL,
    total_debt REAL,
    fico_score INTEGER,
    loan_amount REAL,
    default_flag INTEGER,
    dti_ratio REAL,                    -- Debt-to-Income ratio (calculated)
    fico_category TEXT,                -- FICO score bucket: 'Poor', 'Fair', 'Good', 'Excellent'
    loan_to_income_ratio REAL,         -- Loan amount / Income
    loan_size_category TEXT            -- Loan size: 'Small', 'Medium', 'Large'
);

-- ============================================
-- Table: loan_predictions
-- Purpose: Model predictions (PD values)
-- Source: Python model output
-- ============================================
CREATE TABLE loan_predictions (
    loan_id TEXT PRIMARY KEY,
    pd REAL                            -- Probability of Default (0-1)
);

-- ============================================
-- Table: loan_staging
-- Purpose: IFRS 9 staging and ECL calculation
-- Source: SQL post-processing combining model_features and loan_predictions
-- ============================================
CREATE TABLE loan_staging (
    loan_id TEXT PRIMARY KEY,
    loan_amount REAL,
    default_flag INTEGER,
    pd REAL,                           -- Probability of Default
    stage INTEGER,                     -- IFRS 9 Stage: 1, 2, or 3
    ecl REAL                           -- Expected Credit Loss
);

-- ============================================
-- IFRS 9 Stage Definitions:
-- ============================================
-- Stage 1: Performing loans (PD < 0.02)
--   - 12-month ECL
--   - No significant increase in credit risk
--
-- Stage 2: Significant increase in credit risk (PD >= 0.02)
--   - Lifetime ECL
--   - Credit risk has increased significantly since initial recognition
--
-- Stage 3: Defaulted loans (default_flag = 1)
--   - Lifetime ECL (full expected loss)
--   - Credit-impaired financial asset
--
-- ============================================
-- ECL Calculation Formula:
-- ============================================
-- ECL = PD × LGD × EAD
-- Where:
--   PD  = Probability of Default (from model)
--   LGD = Loss Given Default (assumed 45% in this pipeline)
--   EAD = Exposure at Default (loan_amount)
--
-- Stage-specific adjustments:
--   Stage 1: ECL = EAD × PD × LGD × 0.5 (12-month simplified)
--   Stage 2: ECL = EAD × PD × LGD (lifetime)
--   Stage 3: ECL = EAD × LGD (full expected loss)

