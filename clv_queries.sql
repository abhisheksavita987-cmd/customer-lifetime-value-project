-- =============================================================
--  Predictive Analytics for E-commerce: CLV Forecasting
--  SQL Script  –  MySQL / SQLite Compatible
--  Authors: Abhishek Savita, Aryan Jat, Indrajeet Rawat
--  ITM University, Gwalior
-- =============================================================


-- ─────────────────────────────────────────────────────────────
-- STEP 1 : Create the database  (MySQL only)
-- ─────────────────────────────────────────────────────────────
-- Run in MySQL CLI:
--   CREATE DATABASE ecommerce_clv;
--   USE ecommerce_clv;


-- ─────────────────────────────────────────────────────────────
-- STEP 2 : Create the transactions table
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    id          INT AUTO_INCREMENT PRIMARY KEY,  -- omit in SQLite
    InvoiceNo   VARCHAR(20),
    StockCode   VARCHAR(20),
    Description TEXT,
    Quantity    INT,
    InvoiceDate DATETIME,
    UnitPrice   DECIMAL(10,2),
    CustomerID  VARCHAR(20),
    Country     VARCHAR(50),
    TotalPrice  DECIMAL(12,2)
);

-- NOTE: Python (Pandas + SQLAlchemy / sqlite3) bulk-inserts
--       the cleaned dataframe. No manual INSERT needed.


-- ─────────────────────────────────────────────────────────────
-- STEP 3 : Basic Exploration Queries
-- ─────────────────────────────────────────────────────────────

-- Total number of cleaned transactions
SELECT COUNT(*) AS total_transactions FROM transactions;

-- Unique customers
SELECT COUNT(DISTINCT CustomerID) AS unique_customers FROM transactions;

-- Date range
SELECT MIN(InvoiceDate) AS earliest, MAX(InvoiceDate) AS latest
FROM transactions;

-- Revenue by country (top 10)
SELECT Country,
       ROUND(SUM(TotalPrice), 2) AS total_revenue,
       COUNT(DISTINCT CustomerID) AS customers
FROM   transactions
GROUP  BY Country
ORDER  BY total_revenue DESC
LIMIT  10;


-- ─────────────────────────────────────────────────────────────
-- STEP 4 : RFM Feature Engineering  (pure SQL)
-- ─────────────────────────────────────────────────────────────

-- Reference date = MAX(InvoiceDate) + 1 day
-- Used in Recency calculation

-- MySQL version (uses DATE_ADD and DATEDIFF)
SELECT
    CustomerID,

    -- Recency: days since last purchase
    DATEDIFF(
        DATE_ADD((SELECT MAX(InvoiceDate) FROM transactions), INTERVAL 1 DAY),
        MAX(InvoiceDate)
    ) AS Recency,

    -- Frequency: distinct invoices
    COUNT(DISTINCT InvoiceNo)  AS Frequency,

    -- Monetary: total spend
    ROUND(SUM(TotalPrice), 2)  AS Monetary

FROM transactions
GROUP BY CustomerID
ORDER BY Monetary DESC;


-- ── SQLite-compatible version of the same RFM query ──────────
-- (SQLite uses JULIANDAY instead of DATEDIFF)
SELECT
    CustomerID,
    CAST(
        JULIANDAY((SELECT DATE(MAX(InvoiceDate), '+1 day') FROM transactions))
        - JULIANDAY(MAX(InvoiceDate))
        AS INTEGER
    )                          AS Recency,
    COUNT(DISTINCT InvoiceNo)  AS Frequency,
    ROUND(SUM(TotalPrice), 2)  AS Monetary
FROM transactions
GROUP BY CustomerID
ORDER BY Monetary DESC;


-- ─────────────────────────────────────────────────────────────
-- STEP 5 : (After Python ML)  Store RFM + CLV + Segments
-- ─────────────────────────────────────────────────────────────
-- The Python pipeline writes customer_segments.csv.
-- Load it back into MySQL for Power BI connectivity:

CREATE TABLE IF NOT EXISTS customer_segments (
    CustomerID      VARCHAR(20) PRIMARY KEY,
    Recency         INT,
    Frequency       INT,
    Monetary        DECIMAL(12,2),
    CLV_predicted   DECIMAL(12,2),
    R_score         INT,
    F_score         INT,
    M_score         INT,
    RFM_Score       INT,
    Segment         VARCHAR(20)
);

-- Import via MySQL CLI:
--   LOAD DATA INFILE '/path/to/customer_segments.csv'
--   INTO TABLE customer_segments
--   FIELDS TERMINATED BY ','
--   ENCLOSED BY '"'
--   LINES TERMINATED BY '\n'
--   IGNORE 1 ROWS;


-- ─────────────────────────────────────────────────────────────
-- STEP 6 : Business Intelligence Queries for Power BI
-- ─────────────────────────────────────────────────────────────

-- KPI 1: Total predicted portfolio CLV
SELECT ROUND(SUM(CLV_predicted), 2) AS total_portfolio_clv
FROM customer_segments;

-- KPI 2: Average CLV per customer
SELECT ROUND(AVG(CLV_predicted), 2) AS avg_clv_per_customer
FROM customer_segments;

-- KPI 3: VIP customer count
SELECT COUNT(*) AS vip_count
FROM customer_segments
WHERE Segment = 'VIP';

-- KPI 4: At-Risk customer count
SELECT COUNT(*) AS at_risk_count
FROM customer_segments
WHERE Segment = 'At-Risk';

-- Segment distribution  (for Pie Chart in Power BI)
SELECT Segment,
       COUNT(*) AS customer_count,
       ROUND(AVG(CLV_predicted), 2) AS avg_clv
FROM customer_segments
GROUP BY Segment
ORDER BY customer_count DESC;

-- Top 10 highest CLV customers  (for Bar Chart)
SELECT CustomerID, Segment,
       ROUND(Monetary, 2) AS actual_spend,
       ROUND(CLV_predicted, 2) AS predicted_clv
FROM customer_segments
ORDER BY CLV_predicted DESC
LIMIT 10;

-- RFM score distribution  (for histogram / filters)
SELECT RFM_Score, COUNT(*) AS cnt
FROM customer_segments
GROUP BY RFM_Score
ORDER BY RFM_Score;

-- Monthly revenue trend  (from transactions table)
-- MySQL
SELECT DATE_FORMAT(InvoiceDate, '%Y-%m') AS month,
       ROUND(SUM(TotalPrice), 2)          AS monthly_revenue
FROM transactions
GROUP BY month
ORDER BY month;

-- SQLite
SELECT STRFTIME('%Y-%m', InvoiceDate) AS month,
       ROUND(SUM(TotalPrice), 2)       AS monthly_revenue
FROM transactions
GROUP BY month
ORDER BY month;
