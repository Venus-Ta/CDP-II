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

mapping = pd.read_csv("medicine_id_mapping.csv")
id_to_sku = dict(zip(mapping["medicine_id"], mapping["sku"]))

inv = pd.read_csv("inventory.csv", parse_dates=["snapshot_date", "expiry_date"])
# Keep only RECEIPT rows to avoid duplicates — one row per batch received
receipts = inv[inv["movement_type"] == "RECEIPT"].copy()
receipts = receipts[receipts["medicine_id"].isin(id_to_sku)]

rows = []
for _, row in receipts.iterrows():
    rows.append((
        str(uuid.uuid4()),
        id_to_sku[row["medicine_id"]],
        None,   # supplier_id
        row["batch_number"],
        int(row["movement_quantity"]) if row["movement_quantity"] > 0 else 0,
        int(row["quantity_on_hand"]),
        row["expiry_date"].date(),
        None,   # manufactured_date
        row["snapshot_date"].date(),
        "A1",   # default bin location
        1,
        "available",
        None    # purchase_order_id
    ))

execute_batch(cursor, """
    INSERT INTO stock_batches (
        batch_id, sku, supplier_id, barcode,
        quantity_received, quantity_current,
        expiry_date, manufactured_date, received_date,
        bin_location, shelf_number, batch_status,
        purchase_order_id
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", rows, page_size=500)

conn.commit()
cursor.close()
conn.close()
print(f"Loaded {len(rows)} stock batches.")