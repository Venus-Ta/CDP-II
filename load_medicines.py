import pandas as pd
import psycopg2
import uuid

# 1. Establish the database connection
conn = psycopg2.connect(
    host="127.0.0.1", port=5432,
    dbname="PHARMACY_DB",
    user="postgres", password="407090ST@"
)
cursor = conn.cursor()

# 2. Read the CSV file (This line was missing or misplaced!)
df = pd.read_csv("medicines.csv")

# 3. Loop through rows and insert safely
for _, row in df.iterrows():
    # Convert fields to strings and restrict them to 50 characters maximum
    generic = str(row["generic_name"])[:50]
    brand = str(row["brand_name"])[:50]
    category = str(row["therapeutic_class"])[:50]
    drug_class = str(row["atc_code"])[:50]
    manufacturer = str(row["manufacturer"])[:50]
    dosage = str(row["dosage_form"])[:50]
    strength = str(row["strength"])[:50]

    cursor.execute("""
        INSERT INTO medicines (
            sku, generic_name, brand_name, category,
            drug_class, dosage_form, strength,
            unit_of_measure, shelf_life_days, manufacturer,
            controlled_substance, refrigeration_required
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (
        str(uuid.uuid4()),
        generic,
        brand,
        category,
        drug_class,
        dosage,
        strength,
        "units",
        int(row["shelf_life_days"]),
        manufacturer,
        False,
        row["therapeutic_class"] in ["Antidiabetic"]  # insulin needs refrigeration
    ))

# 4. Commit changes and close connections
conn.commit()
cursor.close()
conn.close()
print("Medicines loaded successfully.")
