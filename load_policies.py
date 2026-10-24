import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import uuid
import math  # Added to check for NaN values

conn = psycopg2.connect(
    host="127.0.0.1", port=5432,
    dbname="PHARMACY_DB",
    user="postgres", password="407090ST@"
)
cursor = conn.cursor()

mapping = pd.read_csv("medicine_id_mapping.csv")
id_to_sku = dict(zip(mapping["medicine_id"], mapping["sku"]))

sales = pd.read_csv("sales.csv", parse_dates=["sale_date"])
purch = pd.read_csv("purchasing.csv")

rows = []
for med_id, sku in id_to_sku.items():
    med_sales = sales[sales["medicine_id"] == med_id]["quantity_sold"]
    med_purch = purch[purch["medicine_id"] == med_id]

    avg_daily   = med_sales.mean() if len(med_sales) > 0 else 1.0
    std_daily   = med_sales.std()  if len(med_sales) > 0 else 0.5
    avg_lead    = med_purch["lead_time_days"].mean() if len(med_purch) > 0 else 7.0

    # Fallback to standard defaults if calculations result in NaN/Null values
    if math.isnan(avg_daily):   avg_daily = 1.0
    if math.isnan(std_daily):   std_daily = 0.5
    if math.isnan(avg_lead):    avg_lead = 7.0

    # Standard formulas
    safety_stock  = round(1.65 * std_daily * (avg_lead ** 0.5), 2)  # 95% service level
    reorder_point = round((avg_daily * avg_lead) + safety_stock, 2)
    eoq           = round(avg_daily * 30, 2)  # 30-day supply as EOQ approximation

    # Explicitly convert to standard Python float/int types to keep psycopg2 clean
    rows.append((
        str(uuid.uuid4()),
        str(sku),
        float(reorder_point),
        float(safety_stock),
        float(eoq),
        0.95,   # 95% service level target
        7,      # weekly review
        0.25,   # 25% annual holding cost rate
        50.0    # fixed ordering cost per order
    ))

execute_batch(cursor, """
    INSERT INTO inventory_policies (
        policy_id, sku, reorder_point, safety_stock_qty,
        eoq_units, service_level_target, review_period_days,
        holding_cost_rate, ordering_cost
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT DO NOTHING
""", rows, page_size=200)

conn.commit()
cursor.close()
conn.close()
print(f"Loaded {len(rows)} inventory policies successfully.")
