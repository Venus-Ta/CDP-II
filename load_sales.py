import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import uuid

conn = psycopg2.connect(
    host="127.0.0.1", port=5432,
    dbname="PHARMACY_DB",
    user="postgres", password="407090ST@"
)
cursor = conn.cursor()

# Load mapping
mapping = pd.read_csv("medicine_id_mapping.csv")
id_to_sku = dict(zip(mapping["medicine_id"], mapping["sku"]))

# Load sales
sales = pd.read_csv("sales.csv", parse_dates=["sale_date"])
sales = sales[sales["medicine_id"].isin(id_to_sku)]

rows = []
for _, row in sales.iterrows():
    rows.append((
        str(uuid.uuid4()),
        id_to_sku[row["medicine_id"]],
        None,   # batch_id — will link after batches loaded
        None,   # pharmacist_id
        None,   # patient_id
        None,   # prescription_id
        int(row["quantity_sold"]),
        float(row["selling_price"]),
        float(row["revenue"]),
        "counter",
        row["sale_date"].to_pydatetime(),
        None    # robot_task_id
    ))

execute_batch(cursor, """
    INSERT INTO sales_transactions (
        transaction_id, sku, batch_id, pharmacist_id,
        patient_id, prescription_id, quantity_dispensed,
        unit_price, total_price, dispense_channel,
        dispensed_at, robot_task_id
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", rows, page_size=1000)

conn.commit()
cursor.close()
conn.close()
print(f"Loaded {len(rows)} sales transactions.")