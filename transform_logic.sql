-- ============================================================================
-- transform_logic.sql
-- Hardcore SQL Script for IFRS 9 ECL Pipeline
-- Demonstrates: CTEs, Window Functions, CASE Logic, DDL/DML/DQL
-- ============================================================================

-- ============================================================================
-- STEP 1: Data Cleaning & Preparation (Landing Zone -> Core Layer)
-- ============================================================================
-- Using CTE (Common Table Expression) for complex transformations
-- Demonstrates: COALESCE, Window Functions, CASE WHEN

DROP TABLE IF EXISTS clean_loans;

CREATE TABLE clean_loans AS
WITH raw_cleaned AS (
    SELECT
        -- Primary Key
        id AS loan_id,
        
        -- Loan Amount (handle NULLs)
        CAST(COALESCE(loan_amnt, 0) AS REAL) AS loan_amount,
        
        -- Employment Length: Parse text to numeric
        -- Examples: '10+ years' -> 10, '< 1 year' -> 0, '5 years' -> 5
        CASE 
            WHEN emp_length IS NULL OR emp_length = 'n/a' THEN NULL
            WHEN emp_length LIKE '%10+%' OR emp_length LIKE '%10+ years%' THEN 10
            WHEN emp_length LIKE '%< 1%' OR emp_length LIKE '%< 1 year%' THEN 0
            WHEN emp_length LIKE '%1 year%' THEN 1
            WHEN emp_length LIKE '%2 years%' THEN 2
            WHEN emp_length LIKE '%3 years%' THEN 3
            WHEN emp_length LIKE '%4 years%' THEN 4
            WHEN emp_length LIKE '%5 years%' THEN 5
            WHEN emp_length LIKE '%6 years%' THEN 6
            WHEN emp_length LIKE '%7 years%' THEN 7
            WHEN emp_length LIKE '%8 years%' THEN 8
            WHEN emp_length LIKE '%9 years%' THEN 9
            ELSE CAST(SUBSTR(emp_length, 1, 1) AS INTEGER)
        END AS emp_years,
        
        -- Annual Income: Use Window Function to fill NULLs with grade-level average
        -- This demonstrates PARTITION BY window function
        -- Note: Window function in COALESCE requires subquery in SQLite
        annual_inc AS annual_income_raw,
        
        -- Credit Grade
        grade,
        sub_grade,
        
        -- Debt-to-Income Ratio (handle NULLs and convert percentage)
        COALESCE(
            CAST(dti AS REAL) / 100.0,
            CASE 
                WHEN annual_inc > 0 AND installment > 0 
                THEN (installment * 12.0) / annual_inc
                ELSE NULL
            END
        ) AS dti_ratio,
        
        -- Loan Status (filter for valid statuses)
        loan_status,
        
        -- FICO Score: Average of low and high range
        CASE 
            WHEN fico_range_low IS NOT NULL AND fico_range_high IS NOT NULL
            THEN (fico_range_low + fico_range_high) / 2.0
            WHEN last_fico_range_low IS NOT NULL AND last_fico_range_high IS NOT NULL
            THEN (last_fico_range_low + last_fico_range_high) / 2.0
            ELSE NULL
        END AS fico_score,
        
        -- Interest Rate (convert percentage string to decimal)
        CAST(REPLACE(int_rate, '%', '') AS REAL) / 100.0 AS interest_rate,
        
        -- Term (extract months)
        CASE 
            WHEN term LIKE '%36%' THEN 36
            WHEN term LIKE '%60%' THEN 60
            ELSE NULL
        END AS term_months,
        
        -- Home Ownership
        home_ownership,
        
        -- Verification Status
        verification_status
        
    FROM raw_landing_zone
    WHERE 
        -- Filter: Only include loans with known outcomes
        loan_status IN ('Fully Paid', 'Charged Off', 'Default', 'Current', 'Late (31-120 days)')
        AND loan_amnt IS NOT NULL
        AND loan_amnt > 0
)
SELECT 
    loan_id,
    loan_amount,
    emp_years,
    -- Fill NULL annual_income with grade-level average using window function
    COALESCE(
        annual_income_raw,
        AVG(annual_income_raw) OVER (PARTITION BY grade)
    ) AS annual_income,
    grade,
    sub_grade,
    dti_ratio,
    loan_status,
    fico_score,
    interest_rate,
    term_months,
    home_ownership,
    verification_status
FROM raw_cleaned
WHERE 
    -- Data quality filters
    annual_income_raw > 0
    AND loan_amount > 0
    AND (fico_score IS NULL OR (fico_score >= 300 AND fico_score <= 850));

-- ============================================================================
-- STEP 2: Feature Engineering (Core Layer -> Model Features)
-- ============================================================================
-- Demonstrates: Business Logic in SQL, CASE WHEN for binning

DROP VIEW IF EXISTS model_features;

CREATE VIEW model_features AS
SELECT
    loan_id,
    loan_amount,
    annual_income,
    COALESCE(dti_ratio, 0) AS dti_ratio,
    COALESCE(fico_score, 680) AS fico_score,  -- Default to median if NULL
    COALESCE(emp_years, 0) AS emp_years,
    term_months,
    interest_rate,
    grade,
    sub_grade,
    home_ownership,
    
    -- Target Variable: Default Flag
    -- Stage 3 (Defaulted) = 1, Others = 0
    CASE 
        WHEN loan_status IN ('Charged Off', 'Default') THEN 1
        WHEN loan_status LIKE 'Late%' THEN 1  -- Late payments considered default
        ELSE 0
    END AS default_flag,
    
    -- Credit Quality Bucket (FICO-based, for WoE binning)
    CASE 
        WHEN fico_score >= 750 THEN 'Excellent'
        WHEN fico_score >= 700 THEN 'Good'
        WHEN fico_score >= 650 THEN 'Fair'
        WHEN fico_score >= 580 THEN 'Poor'
        ELSE 'Very Poor'
    END AS credit_bucket,
    
    -- Loan Size Category
    CASE 
        WHEN loan_amount >= 25000 THEN 'Large'
        WHEN loan_amount >= 10000 THEN 'Medium'
        ELSE 'Small'
    END AS loan_size_category,
    
    -- Income Category (for segmentation)
    CASE 
        WHEN annual_income >= 100000 THEN 'High'
        WHEN annual_income >= 50000 THEN 'Medium'
        ELSE 'Low'
    END AS income_category,
    
    -- DTI Risk Category
    CASE 
        WHEN dti_ratio >= 0.43 THEN 'High Risk'  -- Above 43% is high risk
        WHEN dti_ratio >= 0.36 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS dti_risk_category
    
FROM clean_loans;

-- ============================================================================
-- STEP 3: Model Predictions Table (Created by Python, then enriched by SQL)
-- ============================================================================

DROP TABLE IF EXISTS loan_predictions;

CREATE TABLE loan_predictions (
    loan_id TEXT PRIMARY KEY,
    pd_12m REAL,  -- 12-month Probability of Default
    pd_lifetime REAL,  -- Lifetime Probability of Default
    model_version TEXT DEFAULT 'v1.0'
);

-- ============================================================================
-- STEP 4: IFRS 9 Staging Logic (Post-Modeling)
-- ============================================================================
-- This view will be created AFTER Python writes predictions
-- Demonstrates: Complex CASE WHEN, Subqueries, Window Functions

DROP VIEW IF EXISTS loan_staging;

-- Note: This view will be created after loan_predictions is populated
-- We'll create it in a separate step after modeling

-- ============================================================================
-- STEP 5: ECL Calculation View (Final Output)
-- ============================================================================
-- This will be created after staging is complete
-- Demonstrates: Business calculations in SQL

-- Placeholder for ECL calculation view
-- Will be created after staging logic is applied

