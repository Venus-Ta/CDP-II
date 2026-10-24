from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

# Local database imports
from database import SessionLocal, engine, Base, get_db
# from models import PurchaseOrder

app = FastAPI(
    title="Pharmacy Inventory Management API",
    description="API layer for the autonomous pharmacy robot system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return FileResponse("dashboard.html")

@app.get("/medicines")
def get_medicines(db: Session = Depends(get_db)):
    query = text("""
        SELECT sku::text, generic_name, brand_name,
               category, dosage_form, strength,
               controlled_substance, refrigeration_required
        FROM medicines ORDER BY generic_name ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.get("/inventory/stock-status")
def get_stock_status(db: Session = Depends(get_db)):
    query = text("""
        SELECT m.generic_name, m.dosage_form, m.strength,
               sb.bin_location, sb.quantity_current AS current_stock,
               ip.reorder_point, ip.safety_stock_qty,
               CASE
                   WHEN sb.quantity_current <= ip.safety_stock_qty THEN 'CRITICAL'
                   WHEN sb.quantity_current <= ip.reorder_point THEN 'REORDER NOW'
                   ELSE 'ADEQUATE'
               END AS stock_status
        FROM medicines m
        JOIN stock_batches sb ON sb.sku = m.sku
        JOIN inventory_policies ip ON ip.sku = m.sku
        ORDER BY sb.quantity_current ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.get("/inventory/expiry-risk")
def get_expiry_risk(db: Session = Depends(get_db)):
    query = text("""
        SELECT m.generic_name, sb.barcode, sb.bin_location,
               sb.quantity_current, sb.expiry_date::text,
               (sb.expiry_date - CURRENT_DATE) AS days_until_expiry,
               CASE
                   WHEN (sb.expiry_date - CURRENT_DATE) <= 0  THEN 'EXPIRED'
                   WHEN (sb.expiry_date - CURRENT_DATE) <= 30 THEN 'EXPIRES SOON'
                   WHEN (sb.expiry_date - CURRENT_DATE) <= 60 THEN 'EXPIRING SHORTLY'
                   ELSE 'OK'
               END AS expiry_status
        FROM stock_batches sb
        JOIN medicines m ON m.sku = sb.sku
        WHERE sb.expiry_date <= CURRENT_DATE + INTERVAL '90 days'
        ORDER BY sb.expiry_date ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.get("/inventory/reorder-needed")
def get_reorder_needed(db: Session = Depends(get_db)):
    query = text("""
        SELECT m.generic_name, m.strength,
               sb.quantity_current AS current_stock,
               ip.reorder_point, ip.eoq_units AS recommended_order_qty,
               s.name AS preferred_supplier, s.avg_lead_time_days
        FROM medicines m
        JOIN stock_batches sb ON sb.sku = m.sku
        JOIN inventory_policies ip ON ip.sku = m.sku
        JOIN supplier_medicines sm ON sm.sku = m.sku
        JOIN suppliers s ON s.supplier_id = sm.supplier_id
        WHERE sb.quantity_current <= ip.reorder_point
        ORDER BY sb.quantity_current ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    query = text("""
        SELECT a.alert_id::text, a.alert_type, a.severity,
               m.generic_name, bl.bin_code, a.message,
               a.status, a.created_at::text
        FROM alerts a
        JOIN medicines m ON m.sku = a.sku
        JOIN bin_locations bl ON bl.bin_id = a.bin_id
        WHERE a.status = 'open'
        ORDER BY a.created_at DESC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.get("/suppliers")
def get_suppliers(db: Session = Depends(get_db)):
    query = text("""
        SELECT supplier_id::text, name, contact_email,
               avg_lead_time_days, reliability_score, preferred_supplier
        FROM suppliers ORDER BY name ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

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
        LEFT JOIN medicines m ON m.sku = rt.sku
        LEFT JOIN stock_batches sb ON sb.batch_id = rt.batch_id
        LEFT JOIN bin_locations bl ON bl.bin_id = rt.bin_id
        ORDER BY rt.created_at DESC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

def evaluate_stock_threshold(medicine_name: str):
    """
    Background worker: Evaluates if a dispense event breached the safety threshold.
    """
    db = SessionLocal() 
    
    try:
        print(f"[SYSTEM] Event Triggered: Evaluating inventory threshold for {medicine_name}...")
        
        query = text("""
            WITH stock_calc AS (
                SELECT 
                    m.sku,
                    m.generic_name,
                    COALESCE(SUM(sb.quantity_current), 0) as total_stock
                FROM medicines m
                LEFT JOIN stock_batches sb ON m.sku = sb.sku AND sb.batch_status = 'available'
                WHERE m.generic_name = :medicine_name
                GROUP BY m.sku, m.generic_name
            )
            SELECT 
                sc.sku,
                sc.total_stock,
                ip.reorder_point,
                ip.safety_stock_qty,
                COALESCE(df.predicted_demand, 0) as predicted_demand
            FROM stock_calc sc
            JOIN inventory_policies ip ON sc.sku = ip.sku
            LEFT JOIN demand_forecasts df ON sc.sku = df.sku AND df.forecast_date = CURRENT_DATE
        """)
        
        stock_data = db.execute(query, {"medicine_name": medicine_name}).fetchone()
        
        if not stock_data:
            print(f"[ERROR] Could not find inventory policies for {medicine_name}.")
            return
            
        sku = stock_data._mapping['sku']
        total_stock = stock_data._mapping['total_stock']
        reorder_point = stock_data._mapping['reorder_point']
        safety_stock = stock_data._mapping['safety_stock_qty']
        ai_demand = stock_data._mapping['predicted_demand']
        
        if total_stock <= reorder_point:
            print(f"[ALERT] {medicine_name} stock ({total_stock}) hit the reorder point ({reorder_point})!")
            
            pending_order = db.execute(text("""
                SELECT 1 FROM purchase_orders 
                WHERE medicine_name = :medicine_name AND status = 'PENDING_APPROVAL'
            """), {"medicine_name": medicine_name}).fetchone()
            
            if pending_order:
                print(f"[SYSTEM] Pending order already exists for {medicine_name}. Skipping duplicate.")
                return
                
            order_qty = max(int(ai_demand), int(safety_stock * 2)) 
            
            insert_po = text("""
                INSERT INTO purchase_orders (medicine_name, quantity, status)
                VALUES (:medicine_name, :order_qty, 'PENDING_APPROVAL')
            """)
            db.execute(insert_po, {"medicine_name": medicine_name, "order_qty": order_qty})
            
            db.commit() 
            print(f"✅ [SUCCESS] Auto-Replenishment Triggered: Ordered {order_qty} units of {medicine_name}.")
        else:
            print(f"✓ [OK] {medicine_name} stock is stable at {total_stock}. No order needed.")
            
    except Exception as e:
        print(f"❌ [ERROR] Worker failed: {e}")
        db.rollback()
    finally:
        db.close() 

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
        JOIN stock_batches sb ON sb.sku = m.sku
        JOIN bin_locations bl ON bl.bin_code = sb.bin_location
        WHERE m.generic_name = :medicine_name
        AND sb.batch_status = 'available'
        ORDER BY sb.expiry_date ASC
        LIMIT 1
        RETURNING task_id::text, task_type, status, priority
    """)
    result = db.execute(query, {
        "task_type": task.get("task_type", "PICK"),
        "priority": task.get("priority", 5),
        "quantity": task.get("quantity", 1),
        "medicine_name": task.get("medicine_name")
    }).fetchone()
    db.commit()
    return {"message": "Task created successfully", "task": dict(result._mapping)}

@app.put("/tasks/{task_id}/status")
def update_task_status(task_id: str, update: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    new_status = update.get("status")

    query = text("""
        UPDATE robot_tasks
        SET
            status = :status,
            barcode_scanned = :barcode_scanned,
            barcode_match = :barcode_match,
            quantity_confirmed = :quantity_confirmed,
            completed_at = CASE 
                WHEN :status IN ('completed', 'failed') 
                THEN CURRENT_TIMESTAMP 
                ELSE completed_at 
            END,
            failure_reason = :failure_reason
        WHERE task_id::text = :task_id
    """)
    db.execute(query, {
        "task_id": task_id,
        "status": new_status,
        "barcode_scanned": update.get("barcode_scanned"),
        "barcode_match": update.get("barcode_match"),
        "quantity_confirmed": update.get("quantity_confirmed"),
        "failure_reason": update.get("failure_reason")
    })

    # When a PICK task is completed, decrement stock and re-check threshold
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
            print(f"[STOCK] Decremented {qty} units from batch {task_info.batch_id}")

            if task_info.generic_name:
                background_tasks.add_task(evaluate_stock_threshold, task_info.generic_name)

    db.commit()
    return {
        "message": "Task updated successfully",
        "task_id": task_id,
        "new_status": new_status
    }

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
        LEFT JOIN medicines m ON m.sku = rt.sku
        LEFT JOIN stock_batches sb ON sb.batch_id = rt.batch_id
        LEFT JOIN bin_locations bl ON bl.bin_id = rt.bin_id
        WHERE rt.status = 'queued'
        ORDER BY rt.priority ASC, rt.created_at ASC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.post("/dispense")
def dispense_medicine(request: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    print("\n" + "="*30)
    print("🚀 DISPENSE ENDPOINT ACTIVATED!")
    print(f"Request Data: {request}")
    print("="*30 + "\n")
    print("DEBUG: Dispense endpoint reached!")
    query = text("""
        WITH selected_batch AS (
            SELECT sb.batch_id, sb.sku, sb.bin_location, bl.bin_id,
                   sb.barcode, sb.quantity_current
            FROM stock_batches sb
            JOIN medicines m ON m.sku = sb.sku
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
        JOIN medicines m ON m.sku = nt.sku
    """)
    result = db.execute(query, {
        "medicine_name": request.get("medicine_name"),
        "quantity": request.get("quantity", 1)
    }).fetchone()
    db.commit()
    
    if not result:
        return {"error": "Medicine not found, insufficient stock, or available stock is expired."}
    
    background_tasks.add_task(evaluate_stock_threshold, request["medicine_name"])    
    
    return {
        "message": "Dispense task created",
        "task_id": result._mapping["task_id"],
        "medicine": result._mapping["generic_name"],
        "bin": result._mapping["bin_location"],
        "quantity": result._mapping["quantity_requested"]
    }

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
            sb.quantity_current AS current_stock,
            ip.reorder_point,
            CASE
                WHEN sb.quantity_current < df.predicted_demand * 0.5
                THEN 'URGENT REORDER'
                WHEN sb.quantity_current < df.predicted_demand
                THEN 'REORDER RECOMMENDED'
                ELSE 'STOCK SUFFICIENT'
            END AS ai_recommendation
        FROM demand_forecasts df
        JOIN medicines m ON m.sku = df.sku
        JOIN stock_batches sb ON sb.sku = df.sku
        JOIN inventory_policies ip ON ip.sku = df.sku
        WHERE df.horizon_days = 30
        AND df.forecast_date = CURRENT_DATE
        ORDER BY df.predicted_demand DESC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.get("/forecasts/reorder-recommendations")
def get_ai_reorder_recommendations(db: Session = Depends(get_db)):
    query = text("""
        WITH BatchStock AS (
            -- Aggregate total current stock across all bins/batches for each SKU
            SELECT sku, SUM(quantity_current) AS total_stock
            FROM stock_batches
            GROUP BY sku
        ),
        RankedSuppliers AS (
            -- Select only the fastest preferred supplier per SKU
            SELECT sm.sku, s.name AS supplier, s.avg_lead_time_days,
                   ROW_NUMBER() OVER(PARTITION BY sm.sku ORDER BY s.avg_lead_time_days ASC) as rn
            FROM supplier_medicines sm
            JOIN suppliers s ON s.supplier_id = sm.supplier_id
            WHERE s.preferred_supplier = TRUE
        ),
        LatestForecasts AS (
            -- Isolate only the most recently generated forecast per SKU for today
            SELECT sku, predicted_demand, confidence_upper,
                   ROW_NUMBER() OVER(PARTITION BY sku ORDER BY generated_at DESC) as rn
            FROM demand_forecasts
            WHERE horizon_days = 30 AND forecast_date = CURRENT_DATE
        )
        SELECT
            m.generic_name,
            m.strength,
            df.predicted_demand AS predicted_30_day_demand,
            df.confidence_upper AS worst_case_demand,
            bs.total_stock AS current_stock,
            ip.eoq_units AS recommended_order_qty,
            rs.supplier,
            rs.avg_lead_time_days,
            (df.predicted_demand - bs.total_stock) AS shortfall,
            CASE
                WHEN bs.total_stock < df.predicted_demand * 0.5 THEN 'URGENT — Order immediately'
                WHEN bs.total_stock < df.predicted_demand THEN 'RECOMMENDED — Order within 3 days'
                WHEN bs.total_stock < df.confidence_upper THEN 'MONITOR — May need reorder'
                ELSE 'SUFFICIENT — No action needed'
            END AS action
        FROM LatestForecasts df
        JOIN medicines m ON m.sku = df.sku
        JOIN BatchStock bs ON bs.sku = m.sku
        JOIN inventory_policies ip ON ip.sku = m.sku
        LEFT JOIN RankedSuppliers rs ON rs.sku = m.sku AND rs.rn = 1
        WHERE df.rn = 1
        ORDER BY shortfall DESC
    """)
    results = db.execute(query).fetchall()
    return [dict(row._mapping) for row in results]

@app.get("/robot-tasks")
def get_robot_tasks(limit: int = 50, db: Session = Depends(get_db)):
    tasks = db.execute(text("SELECT * FROM robot_tasks ORDER BY created_at DESC LIMIT :limit"), {"limit": limit}).fetchall()
    return [dict(row._mapping) for row in tasks]

@app.get("/robots/status")
def get_robot_status(db: Session = Depends(get_db)):
    robots = db.execute(text("SELECT * FROM robots")).fetchall()
    return [dict(row._mapping) for row in robots]    

@app.get("/prescriptions/{rx_id}")
def get_prescription(rx_id: str, db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT p.prescription_id, p.patient_id, p.sku, p.quantity_prescribed,
               p.status, p.prescriber_name, p.issued_date,
               pat.patient_code, pat.date_of_birth, pat.allergy_flags,
               m.brand_name, m.generic_name, m.strength, m.dosage_form,
               COALESCE(SUM(sb.quantity_current), 0) as stock_available
        FROM prescriptions p
        JOIN patients pat ON pat.patient_id = p.patient_id
        JOIN medicines m ON m.sku = p.sku
        LEFT JOIN stock_batches sb ON sb.sku = p.sku
        WHERE p.prescription_id = :rx_id OR p.prescription_id::text LIKE :rx_search
        GROUP BY p.prescription_id, p.patient_id, p.sku, p.quantity_prescribed,
                 p.status, p.prescriber_name, p.issued_date,
                 pat.patient_code, pat.date_of_birth, pat.allergy_flags,
                 m.brand_name, m.generic_name, m.strength, m.dosage_form
    """), {"rx_id": rx_id, "rx_search": f"%{rx_id}%"})
    rows = result.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Prescription not found")
    return [dict(row._mapping) for row in rows]

@app.get("/prescriptions")
def list_prescriptions(status: str = "active", db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT p.prescription_id, p.patient_id, p.status, p.issued_date,
               pat.patient_code, pat.allergy_flags,
               m.brand_name, p.quantity_prescribed
        FROM prescriptions p
        JOIN patients pat ON pat.patient_id = p.patient_id
        JOIN medicines m ON m.sku = p.sku
        ORDER BY p.created_at DESC
        LIMIT 20
    """))
    return [dict(row._mapping) for row in result.fetchall()]

class HardwareUpdate(BaseModel):
    status: str
    barcode_scanned: str = None
    failure_reason: str = None

@app.get("/hardware/next-task")
def get_hardware_next_task(db: Session = Depends(get_db)):
    query = text("""
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
    """)
    task = db.execute(query).fetchone()

    if not task:
        return {"command": "IDLE"}

    update_query = text("""
        UPDATE robot_tasks 
        SET status = 'in_progress', started_at = CURRENT_TIMESTAMP 
        WHERE task_id::text = :tid
    """)
    db.execute(update_query, {"tid": task.task_id})
    db.commit()

    return {
        "command": task.task_type,          
        "task_id": task.task_id,
        "target_bin": task.bin_code,        
        "aisle": task.aisle,
        "shelf_number": task.shelf_number,
        "quantity": task.quantity_requested
    }

@app.get("/orders/pending")
def get_pending_orders(db: Session = Depends(get_db)):
    """Fetches all AI-drafted orders that need human approval."""
    # This specifically looks for orders with the 'PENDING_APPROVAL' status
    pending = db.query(PurchaseOrder).filter(PurchaseOrder.status == "PENDING_APPROVAL").all()
    
    # We manually create a list of dictionaries to ensure it is JSON-serializable
    results = []
    for order in pending:
        results.append({
            "order_id": order.order_id,
            "medicine_name": order.medicine_name,
            "quantity": order.quantity,
            "status": order.status,
            "created_at": str(order.created_at) if order.created_at else None,
            "approved_by": order.approved_by
        })
    
    return {
        "status": "success",
        "count": len(results),
        "orders": results
    }

@app.post("/hardware/tasks/{task_id}/complete")
def hardware_complete_task(task_id: str, update: HardwareUpdate, db: Session = Depends(get_db)):
    query = text("""
        UPDATE robot_tasks
        SET
            status = :status,
            barcode_scanned = :barcode_scanned,
            completed_at = CURRENT_TIMESTAMP,
            failure_reason = :failure_reason
        WHERE task_id::text = :task_id
    """)
    db.execute(query, {
        "task_id": task_id,
        "status": update.status, 
        "barcode_scanned": update.barcode_scanned,
        "failure_reason": update.failure_reason
    })
    db.commit()
    return {"status": "success", "message": f"Task {task_id} updated to {update.status}"}