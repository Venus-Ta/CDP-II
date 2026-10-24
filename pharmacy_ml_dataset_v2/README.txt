PHARMACY ML DATASET — FULL PACKAGE
===================================
Generated: October 2026
Coverage : 2018-01-01 to 2024-12-31 (6 years, 2,557 days)
Medicines: 110 SKUs across 26 therapeutic classes
Total rows: 573,254 across 6 tables

FILES
-----
medicines.csv       —  110 rows   — Medicine master (SKU catalog)
suppliers.csv       —    8 rows   — Supplier/distributor records
demand_factors.csv  —  2,557 rows — Daily external signal features
sales.csv           —  281,270 rows — Daily sales transactions per SKU
inventory.csv       —  285,283 rows — Daily stock snapshots + movements
purchasing.csv      —    4,026 rows — Purchase orders + delivery records

KEY COLUMNS
-----------
sales.csv:
  is_stockout_day       → TRUE = stock was zero, demand existed (5.42% of days)
  lost_demand_estimate  → Estimated units lost due to stockout
  price_change_flag     → TRUE = price changed this period
  is_promotion_active   → TRUE = promotional period active

inventory.csv:
  batch_number          → Per-batch tracking (for FEFO logic)
  expiry_date           → Per-batch expiry
  days_to_expiry        → Computed feature for expiry risk model
  expected_waste_units  → Estimated waste if demand < stock before expiry
  movement_type         → RECEIPT / SALE / STOCKOUT / NO_MOVEMENT

demand_factors.csv:
  is_holiday            → UAE public holiday flag
  is_ramadan            → Ramadan period flag
  covid_period_flag     → Mar 2020 – Dec 2021
  disease_trend_index   → Normalized respiratory illness index (0–1)

ML TARGET VARIABLES
-------------------
  Demand forecast   → sales.quantity_sold (predict future)
  Stockout risk     → inventory.stockout_days / sales.is_stockout_day
  Expiry risk       → inventory.expected_waste_units
  Purchase quantity → purchasing.quantity_ordered (recommended)
  Reorder point     → derived from inventory + purchasing tables

VALIDATION RESULTS
------------------
  Stockout rate         : 5.42% (target: 3-8%) ✓
  Negative stock rows   : 0                     ✓
  Received > Ordered    : 0                     ✓
  Full date coverage    : 2018–2024             ✓
  All 110 SKUs present  : Yes                   ✓

LICENSE / USE
-------------
This is a synthetic dataset generated for ML training purposes.
Medicine names, prices, and quantities are simulated.
Real-world demand patterns are based on published pharma sales research.
