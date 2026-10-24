import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="127.0.0.1", port=5432,
    dbname="PHARMACY_DB",
    user="postgres", password="407090ST@"
)

# Get UUID → generic_name from DB
db_meds = pd.read_sql("SELECT sku::text, generic_name, brand_name FROM medicines", conn)

# Get integer ID → generic_name from our CSV
csv_meds = pd.read_csv("medicines.csv")[["medicine_id", "generic_name", "brand_name"]]

# Merge on generic_name
mapping = csv_meds.merge(db_meds, on=["generic_name", "brand_name"], how="inner")
mapping.to_csv("medicine_id_mapping.csv", index=False)
print(f"Mapped {len(mapping)} medicines")
print(mapping.head())
conn.close()