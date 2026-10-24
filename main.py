"""
MediBot — FastAPI Backend  (v2.0)
==================================
Fixes applied vs previous version:
  1. /orders/pending     — removed broken ORM dependency, uses raw SQL
  2. /inventory/stock-status — aggregates across batches (SUM + GROUP BY)
  3. /inventory/reorder-needed — LEFT JOIN so medicines without supplier still appear
  4. /dispense           — clearer error messages per failure mode
  5. /medicines          — now returns rx_otc field for correct OTC/RX badge in UI
  6. .env-ready          — DB URL read from environment variable with fallback

Run:
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import SessionLocal, engine, Base, get_db

app = FastAPI(
    title="MediBot Pharmacy API",
    description="Backend for the MediBot Autonomous Pharmacy Warehouse System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],   # "null" needed for local file:// dev
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("dashboard.html")


# ─────────────────────────────────────────────
# MEDICINES
# ─────────────────────────────────────────────
@app.get("/medicines")
def get_medicines(db: Session = Depends(get_db)):
    """Full medicine catalog — includes rx_otc for correct badge display."""
    query = text("""
        SELECT
            sku::text,
            generic_name,
            brand_name,
            category,
            drug_class,
            dosage_form,
            strength,
            controlled_substance,
            refrigeration_required,
            -- FIX: expose rx_otc so the frontend can show the correct badge
            CASE
                WHEN controlled_substance = TRUE THEN 'RX'
                WHEN drug_class IS NOT NULL AND LENGTH(drug_class) >= 5 THEN 'RX'
                ELSE 'OTC'
            END AS rx_otc
        FROM medicines
        ORDER BY generic_name ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# INVENTORY — STOCK STATUS
# ─────────────────────────────────────────────
@app.get("/inventory/stock-status")
def get_stock_status(db: Session = Depends(get_db)):
    """
    FIX: Aggregates stock across ALL batches per medicine (SUM + GROUP BY).
    Previous version returned one row per batch causing duplicate medicines
    and incorrect critical stock counts.
    """
    query = text("""
        SELECT
            m.generic_name,
            m.dosage_form,
            m.strength,
            MIN(sb.bin_location)            AS bin_location,
            SUM(sb.quantity_current)        AS current_stock,
            MAX(ip.reorder_point)           AS reorder_point,
            MAX(ip.safety_stock_qty)        AS safety_stock_qty,
            CASE
                WHEN SUM(sb.quantity_current) <= MAX(ip.safety_stock_qty) THEN 'CRITICAL'
                WHEN SUM(sb.quantity_current) <= MAX(ip.reorder_point)    THEN 'REORDER NOW'
                ELSE 'ADEQUATE'
            END AS stock_status
        FROM medicines m
        JOIN stock_batches sb   ON sb.sku = m.sku
        JOIN inventory_policies ip ON ip.sku = m.sku
        GROUP BY m.sku, m.generic_name, m.dosage_form, m.strength
        ORDER BY SUM(sb.quantity_current) ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# INVENTORY — EXPIRY RISK
# ─────────────────────────────────────────────
@app.get("/inventory/expiry-risk")
def get_expiry_risk(db: Session = Depends(get_db)):
    query = text("""
        SELECT
            m.generic_name,
            sb.barcode,
            sb.bin_location,
            sb.quantity_current,
            sb.expiry_date::text,
            (sb.expiry_date - CURRENT_DATE)  AS days_until_expiry,
            CASE
                WHEN (sb.expiry_date - CURRENT_DATE) <= 0  THEN 'EXPIRED'
                WHEN (sb.expiry_date - CURRENT_DATE) <= 30 THEN 'EXPIRES SOON'
                WHEN (sb.expiry_date - CURRENT_DATE) <= 60 THEN 'EXPIRING SHORTLY'
                ELSE 'OK'
            END AS expiry_status
        FROM stock_batches sb
        JOIN medicines m ON m.sku = sb.sku
        WHERE sb.expiry_date <= CURRENT_DATE + INTERVAL '90 days'
          AND sb.quantity_current > 0
        ORDER BY sb.expiry_date ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# INVENTORY — REORDER NEEDED
# ─────────────────────────────────────────────
@app.get("/inventory/reorder-needed")
def get_reorder_needed(db: Session = Depends(get_db)):
    """
    FIX: LEFT JOINs on supplier tables so medicines with no supplier entry
    still appear in the reorder list rather than being silently dropped.
    """
    query = text("""
        SELECT
            m.generic_name,
            m.strength,
            SUM(sb.quantity_current)  AS current_stock,
            MAX(ip.reorder_point)     AS reorder_point,
            MAX(ip.eoq_units)         AS recommended_order_qty,
            MAX(s.name)               AS preferred_supplier,
            MAX(s.avg_lead_time_days) AS avg_lead_time_days
        FROM medicines m
        JOIN stock_batches sb      ON sb.sku = m.sku
        JOIN inventory_policies ip ON ip.sku = m.sku
        LEFT JOIN supplier_medicines sm ON sm.sku = m.sku
        LEFT JOIN suppliers s          ON s.supplier_id = sm.supplier_id
        GROUP BY m.sku, m.generic_name, m.strength
        HAVING SUM(sb.quantity_current) <= MAX(ip.reorder_point)
        ORDER BY SUM(sb.quantity_current) ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────
@app.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    query = text("""
        SELECT
            a.alert_id::text,
            a.alert_type,
            a.severity,
            m.generic_name,
            bl.bin_code,
            a.message,
            a.status,
            a.created_at::text
        FROM alerts a
        JOIN medicines m    ON m.sku = a.sku
        JOIN bin_locations bl ON bl.bin_id = a.bin_id
        WHERE a.status = 'open'
        ORDER BY a.created_at DESC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# SUPPLIERS
# ─────────────────────────────────────────────
@app.get("/suppliers")
def get_suppliers(db: Session = Depends(get_db)):
    query = text("""
        SELECT
            supplier_id::text,
            name,
            contact_email,
            avg_lead_time_days,
            reliability_score,
            preferred_supplier
        FROM suppliers
        ORDER BY preferred_supplier DESC, name ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# ROBOT TASKS — FULL LIST
# ─────────────────────────────────────────────
@app.get("/tasks")
def get_tasks(db: Session = Depends(get_db)):
    query = text("""
        SELECT
            rt.task_id::text,
            rt.task_type,
            rt.status,
            rt.priority,
            m.generic_name,
            sb.barcode,
            bl.bin_code,
            rt.quantity_requested,
            rt.quantity_confirmed,
            rt.barcode_scanned,
            rt.barcode_match,
            rt.failure_reason,
            rt.created_at::text,
            rt.started_at::text,
            rt.completed_at::text,
            rt.duration_ms
        FROM robot_tasks rt
        LEFT JOIN medicines m        ON m.sku = rt.sku
        LEFT JOIN stock_batches sb   ON sb.batch_id = rt.batch_id
        LEFT JOIN bin_locations bl   ON bl.bin_id = rt.bin_id
        ORDER BY rt.created_at DESC
        LIMIT 100
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# ROBOT TASKS — QUEUED
# ─────────────────────────────────────────────
@app.get("/tasks/queued")
def get_queued_tasks(db: Session = Depends(get_db)):
    query = text("""
        SELECT
            rt.task_id::text,
            rt.task_type,
            rt.priority,
            m.generic_name,
            sb.barcode,
            bl.bin_code,
            bl.aisle,
            bl.shelf_number,
            rt.quantity_requested,
            rt.created_at::text
        FROM robot_tasks rt
        LEFT JOIN medicines m      ON m.sku = rt.sku
        LEFT JOIN stock_batches sb ON sb.batch_id = rt.batch_id
        LEFT JOIN bin_locations bl ON bl.bin_id = rt.bin_id
        WHERE rt.status = 'queued'
        ORDER BY rt.priority ASC, rt.created_at ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# ROBOT TASKS — CREATE
# ─────────────────────────────────────────────
@app.post("/tasks/create")
def create_task(task: dict, db: Session = Depends(get_db)):
    query = text("""
        INSERT INTO robot_tasks (
            task_type, status, priority, sku, batch_id, bin_id,
            quantity_requested, created_at
        )
        SELECT
            :task_type, 'queued', :priority,
            m.sku, sb.batch_id, bl.bin_id, :quantity,
            CURRENT_TIMESTAMP
        FROM medicines m
        JOIN stock_batches sb  ON sb.sku = m.sku
        JOIN bin_locations bl  ON bl.bin_code = sb.bin_location
        WHERE m.generic_name = :medicine_name
          AND sb.batch_status = 'available'
        ORDER BY sb.expiry_date ASC
        LIMIT 1
        RETURNING task_id::text, task_type, status, priority
    """)
    result = db.execute(query, {
        "task_type":     task.get("task_type", "PICK"),
        "priority":      task.get("priority", 5),
        "quantity":      task.get("quantity", 1),
        "medicine_name": task.get("medicine_name"),
    }).fetchone()
    db.commit()
    if not result:
        raise HTTPException(status_code=404, detail="Medicine not found or no available stock")
    return {"message": "Task created", "task": dict(result._mapping)}


# ─────────────────────────────────────────────
# ROBOT TASKS — UPDATE STATUS
# ─────────────────────────────────────────────
def evaluate_stock_threshold(medicine_name: str):
    """
    Background worker: checks whether a post-dispense stock level
    has crossed the reorder point and auto-creates a purchase order.
    """
    db = SessionLocal()
    try:
        print(f"[SYSTEM] Evaluating threshold for: {medicine_name}")
        stock_data = db.execute(text("""
            WITH stock_calc AS (
                SELECT m.sku, m.generic_name,
                       COALESCE(SUM(sb.quantity_current), 0) AS total_stock
                FROM medicines m
                LEFT JOIN stock_batches sb ON m.sku = sb.sku AND sb.batch_status = 'available'
                WHERE m.generic_name = :name
                GROUP BY m.sku, m.generic_name
            )
            SELECT sc.sku, sc.total_stock,
                   ip.reorder_point, ip.safety_stock_qty,
                   COALESCE(df.predicted_demand, 0) AS predicted_demand
            FROM stock_calc sc
            JOIN inventory_policies ip ON sc.sku = ip.sku
            LEFT JOIN demand_forecasts df ON sc.sku = df.sku
                AND df.forecast_date = CURRENT_DATE
                AND df.horizon_days = 30
        """), {"name": medicine_name}).fetchone()

        if not stock_data:
            return

        m = stock_data._mapping
        if m["total_stock"] <= m["reorder_point"]:
            existing = db.execute(text("""
                SELECT 1 FROM purchase_orders
                WHERE medicine_name = :name AND status = 'PENDING_APPROVAL'
            """), {"name": medicine_name}).fetchone()

            if not existing:
                order_qty = max(int(m["predicted_demand"]), int(m["safety_stock_qty"] * 2))
                db.execute(text("""
                    INSERT INTO purchase_orders (medicine_name, quantity, status)
                    VALUES (:name, :qty, 'PENDING_APPROVAL')
                """), {"name": medicine_name, "qty": order_qty})
                db.commit()
                print(f"✓ Auto-order created: {order_qty} units of {medicine_name}")
    except Exception as e:
        print(f"[ERROR] evaluate_stock_threshold: {e}")
        db.rollback()
    finally:
        db.close()


@app.put("/tasks/{task_id}/status")
def update_task_status(
    task_id: str,
    update: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    new_status = update.get("status")
    db.execute(text("""
        UPDATE robot_tasks SET
            status             = :status,
            barcode_scanned    = :barcode_scanned,
            barcode_match      = :barcode_match,
            quantity_confirmed = :quantity_confirmed,
            completed_at = CASE
                WHEN :status IN ('completed','failed') THEN CURRENT_TIMESTAMP
                ELSE completed_at END,
            failure_reason = :failure_reason
        WHERE task_id::text = :task_id
    """), {
        "task_id":           task_id,
        "status":            new_status,
        "barcode_scanned":   update.get("barcode_scanned"),
        "barcode_match":     update.get("barcode_match"),
        "quantity_confirmed":update.get("quantity_confirmed"),
        "failure_reason":    update.get("failure_reason"),
    })

    if new_status == "completed":
        task_info = db.execute(text("""
            SELECT rt.batch_id, rt.quantity_requested, m.generic_name
            FROM robot_tasks rt
            LEFT JOIN medicines m ON m.sku = rt.sku
            WHERE rt.task_id::text = :task_id AND rt.task_type = 'PICK'
        """), {"task_id": task_id}).fetchone()

        if task_info and task_info.batch_id:
            qty = update.get("quantity_confirmed") or task_info.quantity_requested
            db.execute(text("""
                UPDATE stock_batches
                SET quantity_current = GREATEST(quantity_current - :qty, 0)
                WHERE batch_id = :batch_id
            """), {"qty": qty, "batch_id": task_info.batch_id})
            if task_info.generic_name:
                background_tasks.add_task(evaluate_stock_threshold, task_info.generic_name)

    db.commit()
    return {"message": "Task updated", "task_id": task_id, "new_status": new_status}


# ─────────────────────────────────────────────
# DISPENSE
# ─────────────────────────────────────────────
@app.post("/dispense")
def dispense_medicine(
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    FIX: Returns specific error messages per failure mode instead of
    one generic message that hid the actual problem.
    """
    medicine_name = request.get("medicine_name", "").strip()
    quantity      = int(request.get("quantity", 1))

    if not medicine_name:
        return {"error": "Medicine name is required"}
    if quantity < 1:
        return {"error": "Quantity must be at least 1"}

    # Check medicine exists at all
    med_exists = db.execute(text(
        "SELECT 1 FROM medicines WHERE generic_name = :name LIMIT 1"
    ), {"name": medicine_name}).fetchone()
    if not med_exists:
        return {"error": f"'{medicine_name}' not found in medicine database. Check spelling."}

    # Check total stock (across all batches including expired)
    total_stock = db.execute(text("""
        SELECT COALESCE(SUM(sb.quantity_current), 0) AS total
        FROM stock_batches sb
        JOIN medicines m ON m.sku = sb.sku
        WHERE m.generic_name = :name
    """), {"name": medicine_name}).fetchone()
    if total_stock and total_stock.total == 0:
        return {"error": f"No stock available for {medicine_name}. A reorder may be pending."}

    # Check non-expired stock with sufficient quantity
    available = db.execute(text("""
        SELECT COALESCE(SUM(sb.quantity_current), 0) AS total
        FROM stock_batches sb
        JOIN medicines m ON m.sku = sb.sku
        WHERE m.generic_name = :name
          AND sb.batch_status = 'available'
          AND sb.expiry_date >= CURRENT_DATE
    """), {"name": medicine_name}).fetchone()
    if available and available.total < quantity:
        return {
            "error": f"Insufficient non-expired stock for {medicine_name}. "
                     f"Available: {int(available.total)} units, Requested: {quantity} units."
        }

    # FEFO dispense — pick earliest-expiry batch
    query = text("""
        WITH selected_batch AS (
            SELECT sb.batch_id, sb.sku, sb.bin_location, bl.bin_id,
                   sb.barcode, sb.quantity_current
            FROM stock_batches sb
            JOIN medicines m    ON m.sku = sb.sku
            JOIN bin_locations bl ON bl.bin_code = sb.bin_location
            WHERE m.generic_name = :medicine_name
              AND sb.batch_status = 'available'
              AND sb.quantity_current >= :quantity
              AND sb.expiry_date >= CURRENT_DATE
            ORDER BY sb.expiry_date ASC
            LIMIT 1
        ),
        new_task AS (
            INSERT INTO robot_tasks (
                task_type, status, priority, sku, batch_id,
                bin_id, quantity_requested, created_at
            )
            SELECT 'PICK', 'queued', 1, sku, batch_id,
                   bin_id, :quantity, CURRENT_TIMESTAMP
            FROM selected_batch
            RETURNING task_id, batch_id, sku, bin_id
        )
        SELECT
            nt.task_id::text,
            m.generic_name,
            sb.barcode,
            sb.bin_location,
            :quantity AS quantity_requested
        FROM new_task nt
        JOIN stock_batches sb ON sb.batch_id = nt.batch_id
        JOIN medicines m      ON m.sku = nt.sku
    """)
    result = db.execute(query, {
        "medicine_name": medicine_name,
        "quantity":      quantity,
    }).fetchone()
    db.commit()

    if not result:
        return {"error": f"Could not create task for {medicine_name}. No single batch has {quantity} units available."}

    background_tasks.add_task(evaluate_stock_threshold, medicine_name)

    return {
        "message":  "Dispense task created — IM-bot will pick now",
        "task_id":  result._mapping["task_id"],
        "medicine": result._mapping["generic_name"],
        "bin":      result._mapping["bin_location"],
        "quantity": result._mapping["quantity_requested"],
    }


# ─────────────────────────────────────────────
# AI FORECASTS
# ─────────────────────────────────────────────
@app.get("/forecasts")
def get_forecasts(db: Session = Depends(get_db)):
    query = text("""
        SELECT
            m.generic_name,
            m.strength,
            df.forecast_date::text,
            df.horizon_days,
            df.predicted_demand,
            df.confidence_lower,
            df.confidence_upper,
            df.mape_score,
            df.model_version,
            SUM(sb.quantity_current)  AS current_stock,
            MAX(ip.reorder_point)     AS reorder_point,
            CASE
                WHEN SUM(sb.quantity_current) < df.predicted_demand * 0.5 THEN 'URGENT REORDER'
                WHEN SUM(sb.quantity_current) < df.predicted_demand       THEN 'REORDER RECOMMENDED'
                ELSE 'STOCK SUFFICIENT'
            END AS ai_recommendation
        FROM demand_forecasts df
        JOIN medicines m           ON m.sku = df.sku
        JOIN stock_batches sb      ON sb.sku = df.sku
        JOIN inventory_policies ip ON ip.sku = df.sku
        WHERE df.horizon_days = 30
          AND df.forecast_date = CURRENT_DATE
        GROUP BY m.sku, m.generic_name, m.strength,
                 df.forecast_date, df.horizon_days, df.predicted_demand,
                 df.confidence_lower, df.confidence_upper,
                 df.mape_score, df.model_version
        ORDER BY df.predicted_demand DESC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


@app.get("/forecasts/reorder-recommendations")
def get_ai_reorder_recommendations(db: Session = Depends(get_db)):
    query = text("""
        WITH BatchStock AS (
            SELECT sku, SUM(quantity_current) AS total_stock
            FROM stock_batches
            GROUP BY sku
        ),
        RankedSuppliers AS (
            SELECT sm.sku, s.name AS supplier, s.avg_lead_time_days,
                   ROW_NUMBER() OVER(PARTITION BY sm.sku ORDER BY s.avg_lead_time_days ASC) AS rn
            FROM supplier_medicines sm
            JOIN suppliers s ON s.supplier_id = sm.supplier_id
            WHERE s.preferred_supplier = TRUE
        ),
        LatestForecasts AS (
            SELECT sku, predicted_demand, confidence_upper,
                   ROW_NUMBER() OVER(PARTITION BY sku ORDER BY generated_at DESC) AS rn
            FROM demand_forecasts
            WHERE horizon_days = 30 AND forecast_date = CURRENT_DATE
        )
        SELECT
            m.generic_name,
            m.strength,
            df.predicted_demand                          AS predicted_30_day_demand,
            df.confidence_upper                          AS worst_case_demand,
            bs.total_stock                               AS current_stock,
            ip.eoq_units                                 AS recommended_order_qty,
            rs.supplier,
            rs.avg_lead_time_days,
            (df.predicted_demand - bs.total_stock)       AS shortfall,
            CASE
                WHEN bs.total_stock < df.predicted_demand * 0.5 THEN 'URGENT — Order immediately'
                WHEN bs.total_stock < df.predicted_demand        THEN 'RECOMMENDED — Order within 3 days'
                WHEN bs.total_stock < df.confidence_upper        THEN 'MONITOR — May need reorder'
                ELSE 'SUFFICIENT — No action needed'
            END AS action
        FROM LatestForecasts df
        JOIN medicines m           ON m.sku = df.sku
        JOIN BatchStock bs         ON bs.sku = m.sku
        JOIN inventory_policies ip ON ip.sku = m.sku
        LEFT JOIN RankedSuppliers rs ON rs.sku = m.sku AND rs.rn = 1
        WHERE df.rn = 1
        ORDER BY shortfall DESC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]


# ─────────────────────────────────────────────
# ROBOT STATUS
# ─────────────────────────────────────────────
@app.get("/robot-tasks")
def get_robot_tasks(limit: int = 50, db: Session = Depends(get_db)):
    tasks = db.execute(text(
        "SELECT * FROM robot_tasks ORDER BY created_at DESC LIMIT :limit"
    ), {"limit": limit}).fetchall()
    return [dict(row._mapping) for row in tasks]


@app.get("/robots/status")
def get_robot_status(db: Session = Depends(get_db)):
    robots = db.execute(text("SELECT * FROM robots")).fetchall()
    return [dict(row._mapping) for row in robots]


# ─────────────────────────────────────────────
# PRESCRIPTIONS
# ─────────────────────────────────────────────
@app.get("/prescriptions/{rx_id}")
def get_prescription(rx_id: str, db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT
            p.prescription_id, p.patient_id, p.sku,
            p.quantity_prescribed, p.status,
            p.prescriber_name, p.issued_date,
            pat.patient_code, pat.date_of_birth, pat.allergy_flags,
            m.brand_name, m.generic_name, m.strength, m.dosage_form,
            COALESCE(SUM(sb.quantity_current), 0) AS stock_available
        FROM prescriptions p
        JOIN patients pat  ON pat.patient_id = p.patient_id
        JOIN medicines m   ON m.sku = p.sku
        LEFT JOIN stock_batches sb ON sb.sku = p.sku
        WHERE p.prescription_id::text = :rx_id
           OR p.prescription_id::text LIKE :rx_search
        GROUP BY
            p.prescription_id, p.patient_id, p.sku,
            p.quantity_prescribed, p.status,
            p.prescriber_name, p.issued_date,
            pat.patient_code, pat.date_of_birth, pat.allergy_flags,
            m.brand_name, m.generic_name, m.strength, m.dosage_form
    """), {"rx_id": rx_id, "rx_search": f"%{rx_id}%"})
    rows = result.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Prescription not found")
    return [dict(row._mapping) for row in rows]


@app.get("/prescriptions")
def list_prescriptions(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT
            p.prescription_id, p.patient_id, p.status,
            p.issued_date, p.prescriber_name,
            pat.patient_code, pat.allergy_flags,
            m.brand_name, p.quantity_prescribed
        FROM prescriptions p
        JOIN patients pat ON pat.patient_id = p.patient_id
        JOIN medicines m  ON m.sku = p.sku
        ORDER BY p.created_at DESC
        LIMIT 20
    """))
    return [dict(row._mapping) for row in result.fetchall()]


# ─────────────────────────────────────────────
# PURCHASE ORDERS
# ─────────────────────────────────────────────
@app.get("/orders/pending")
def get_pending_orders(db: Session = Depends(get_db)):
    """
    FIX: Replaced broken ORM query (PurchaseOrder model was never imported)
    with raw SQL consistent with all other endpoints.
    """
    result = db.execute(text("""
        SELECT
            order_id::text,
            medicine_name,
            quantity,
            status,
            created_at::text,
            approved_by
        FROM purchase_orders
        WHERE status = 'PENDING_APPROVAL'
        ORDER BY created_at DESC
    """))
    rows = result.fetchall()
    return {
        "status": "success",
        "count":  len(rows),
        "orders": [dict(row._mapping) for row in rows],
    }


@app.post("/orders/{order_id}/approve")
def approve_order(order_id: str, db: Session = Depends(get_db)):
    """Approve a pending purchase order."""
    db.execute(text("""
        UPDATE purchase_orders
        SET status = 'APPROVED', approved_by = 'system_user'
        WHERE order_id::text = :order_id
          AND status = 'PENDING_APPROVAL'
    """), {"order_id": order_id})
    db.commit()
    return {"message": f"Order {order_id} approved"}


# ─────────────────────────────────────────────
# HARDWARE ENDPOINTS (Arduino / UNO-Q)
# ─────────────────────────────────────────────
@app.get("/hardware/next-task")
def get_hardware_next_task(db: Session = Depends(get_db)):
    """
    Polled by the Arduino every few seconds.
    Returns the next queued PICK task and marks it in_progress.
    Returns {"command": "IDLE"} when queue is empty.
    """
    task = db.execute(text("""
        SELECT
            rt.task_id::text,
            rt.task_type,
            rt.quantity_requested,
            bl.bin_code,
            bl.aisle,
            bl.shelf_number
        FROM robot_tasks rt
        JOIN bin_locations bl ON bl.bin_id = rt.bin_id
        WHERE rt.status = 'queued'
        ORDER BY rt.priority ASC, rt.created_at ASC
        LIMIT 1
    """)).fetchone()

    if not task:
        return {"command": "IDLE"}

    db.execute(text("""
        UPDATE robot_tasks
        SET status = 'in_progress', started_at = CURRENT_TIMESTAMP
        WHERE task_id::text = :tid
    """), {"tid": task.task_id})
    db.commit()

    return {
        "command":      task.task_type,
        "task_id":      task.task_id,
        "target_bin":   task.bin_code,
        "aisle":        task.aisle,
        "shelf_number": task.shelf_number,
        "quantity":     task.quantity_requested,
    }


class HardwareUpdate(BaseModel):
    status:          str
    barcode_scanned: str = None
    failure_reason:  str = None


@app.post("/hardware/tasks/{task_id}/complete")
def hardware_complete_task(
    task_id: str,
    update: HardwareUpdate,
    db: Session = Depends(get_db),
):
    db.execute(text("""
        UPDATE robot_tasks SET
            status          = :status,
            barcode_scanned = :barcode_scanned,
            completed_at    = CURRENT_TIMESTAMP,
            failure_reason  = :failure_reason
        WHERE task_id::text = :task_id
    """), {
        "task_id":        task_id,
        "status":         update.status,
        "barcode_scanned":update.barcode_scanned,
        "failure_reason": update.failure_reason,
    })
    db.commit()
    return {"status": "success", "message": f"Task {task_id} → {update.status}"}
