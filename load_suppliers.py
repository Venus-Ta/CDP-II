import pandas as pd
import psycopg2
import uuid

conn = psycopg2.connect(
    host="127.0.0.1", port=5432,
    dbname="PHARMACY_DB",
    user="postgres", password="407090ST@"
)
cursor = conn.cursor()

df = pd.read_csv("suppliers.csv")

for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO suppliers (
            supplier_id, name, avg_lead_time_days,
            reliability_score, preferred_supplier
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (
        str(uuid.uuid4()),
        row["supplier_name"],
        int(row["avg_lead_time_days"]),
        float(row["delivery_reliability_score"]),
        row["delivery_reliability_score"] >= 0.92
    ))

conn.commit()
cursor.close()
conn.close()
print("Suppliers loaded.")