"""
MediBot — AI Demand Forecasting Model  (v2.1)
==============================================
Architecture  : Single global LightGBM model (all medicines, all dates)
Features      : Lag values, rolling statistics, calendar signals,
                external demand factors (holiday, Ramadan, COVID,
                promotion, disease trend, price change)
Evaluation    : Time-based train / validation / test split (no data leakage)
Persistence   : Model saved to  models/lgbm_demand_model.pkl  via joblib
Output        : Forecasts written to  demand_forecasts  table in PostgreSQL
Horizons      : 7 / 14 / 30 days ahead per medicine

Fixes in v2.1:
  - Transaction rollback after any failed query (prevents InFailedSqlTransaction)
  - demand_factors table is optional — graceful fallback to calendar-only features
  - stock_batches / inventory_policies / suppliers all fail gracefully
  - Each DB load wrapped in its own try/except with rollback

Run:
    python forecasting_model.py

Requirements:
    pip install lightgbm joblib scikit-learn psycopg2-binary pandas numpy
"""

import os
import warnings
import psycopg2
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DB_CONFIG = dict(
    host     = "127.0.0.1",
    port     = 5432,
    dbname   = "PHARMACY_DB",
    user     = "postgres",
    password = "407090ST@",
)

MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_demand_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

HORIZONS       = [7, 14, 30]
TEST_DAYS      = 30
VAL_DAYS       = 30
MIN_HISTORY    = 60
CONFIDENCE_PCT = 0.20

LGBM_PARAMS = dict(
    objective         = "regression_l1",
    metric            = "mape",
    learning_rate     = 0.05,
    n_estimators      = 600,
    num_leaves        = 63,
    max_depth         = 7,
    min_child_samples = 20,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    verbose           = -1,
    n_jobs            = -1,
)

# ─────────────────────────────────────────────────────────
# DATABASE — safe query helper
# ─────────────────────────────────────────────────────────
def connect():
    print("Connecting to PHARMACY_DB …")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    print("✓ Connected\n")
    return conn


def safe_query(conn, sql, label="query") -> pd.DataFrame:
    """
    Run a SELECT and return a DataFrame.
    On ANY error: rollback the transaction so subsequent queries work,
    print a warning, and return an empty DataFrame.
    Never raises — the caller always gets a DataFrame back.
    """
    try:
        df = pd.read_sql(sql, conn)
        conn.commit()          # end transaction cleanly
        return df
    except Exception as e:
        try:
            conn.rollback()    # CRITICAL — resets the failed transaction
        except Exception:
            pass
        print(f"  ⚠  Could not load {label}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
def load_sales(conn) -> pd.DataFrame:
    sql = """
        SELECT
            m.sku::text                  AS sku,
            m.generic_name,
            m.category                   AS therapeutic_class,
            DATE(st.dispensed_at)        AS sale_date,
            SUM(st.quantity_dispensed)   AS quantity_sold,
            AVG(st.unit_price)           AS avg_selling_price
        FROM sales_transactions st
        JOIN medicines m ON m.sku = st.sku
        GROUP BY m.sku, m.generic_name, m.category, DATE(st.dispensed_at)
        ORDER BY m.generic_name, sale_date
    """
    df = safe_query(conn, sql, "sales_transactions")
    if df.empty:
        return df
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    print(f"Loaded {len(df):,} daily sales records across "
          f"{df['generic_name'].nunique()} medicines "
          f"({df['sale_date'].min().date()} → {df['sale_date'].max().date()})")
    return df


def load_demand_factors(conn) -> pd.DataFrame:
    sql = """
        SELECT date, is_holiday, is_promotion_active, is_ramadan,
               covid_period_flag, disease_trend_index,
               day_of_week, month, is_weekend
        FROM demand_factors
        ORDER BY date
    """
    df = safe_query(conn, sql, "demand_factors")
    if df.empty:
        print("  demand_factors not available — using calendar-only features")
        return df
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} demand factor rows")
    return df


def load_current_stock(conn) -> pd.DataFrame:
    sql = """
        SELECT sku::text, SUM(quantity_current) AS current_stock
        FROM stock_batches
        GROUP BY sku
    """
    df = safe_query(conn, sql, "stock_batches")
    if not df.empty:
        print(f"Loaded stock for {len(df)} medicines")
    return df


def load_inventory_policies(conn) -> pd.DataFrame:
    sql = """
        SELECT sku::text, reorder_point, safety_stock_qty, eoq_units
        FROM inventory_policies
    """
    df = safe_query(conn, sql, "inventory_policies")
    if not df.empty:
        print(f"Loaded inventory policies for {len(df)} medicines")
    return df


def load_suppliers(conn) -> pd.DataFrame:
    sql = """
        SELECT sm.sku::text, s.name AS supplier, s.avg_lead_time_days
        FROM supplier_medicines sm
        JOIN suppliers s ON s.supplier_id = sm.supplier_id
        WHERE s.preferred_supplier = TRUE
    """
    df = safe_query(conn, sql, "suppliers")
    return df


# ─────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
def build_features(df_all: pd.DataFrame, demand_factors: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding features …")

    le       = LabelEncoder()
    le_class = LabelEncoder()
    df_all["medicine_id"] = le.fit_transform(df_all["generic_name"])
    df_all["class_id"]    = le_class.fit_transform(df_all["therapeutic_class"].fillna("Unknown"))

    frames = []
    skipped = 0

    for med_id in df_all["medicine_id"].unique():
        df_med = df_all[df_all["medicine_id"] == med_id].copy()
        df_med = df_med.sort_values("sale_date").reset_index(drop=True)

        if len(df_med) < MIN_HISTORY:
            skipped += 1
            continue

        # ── Calendar features ────────────────────────────
        df_med["day_of_week"]    = df_med["sale_date"].dt.dayofweek
        df_med["month"]          = df_med["sale_date"].dt.month
        df_med["day_of_month"]   = df_med["sale_date"].dt.day
        df_med["week_of_year"]   = df_med["sale_date"].dt.isocalendar().week.astype(int)
        df_med["quarter"]        = df_med["sale_date"].dt.quarter
        df_med["is_weekend"]     = (df_med["day_of_week"] >= 4).astype(int)
        df_med["is_month_start"] = (df_med["day_of_month"] <= 3).astype(int)
        df_med["is_month_end"]   = (df_med["day_of_month"] >= 28).astype(int)
        df_med["month_sin"]      = np.sin(2 * np.pi * df_med["month"] / 12)
        df_med["month_cos"]      = np.cos(2 * np.pi * df_med["month"] / 12)
        df_med["dow_sin"]        = np.sin(2 * np.pi * df_med["day_of_week"] / 7)
        df_med["dow_cos"]        = np.cos(2 * np.pi * df_med["day_of_week"] / 7)
        df_med["is_winter"]      = df_med["month"].isin([1,2,3,10,11,12]).astype(int)

        # ── Lag features ─────────────────────────────────
        for lag in [1, 2, 3, 7, 14, 21, 28, 60, 90]:
            df_med[f"lag_{lag}"] = df_med["quantity_sold"].shift(lag)

        # ── Rolling statistics ────────────────────────────
        for window in [7, 14, 30, 60, 90]:
            shifted = df_med["quantity_sold"].shift(1)
            rolled  = shifted.rolling(window, min_periods=max(3, window // 3))
            df_med[f"roll_mean_{window}"] = rolled.mean()
            df_med[f"roll_std_{window}"]  = rolled.std()
            df_med[f"roll_max_{window}"]  = rolled.max()
            df_med[f"roll_min_{window}"]  = rolled.min()

        # Demand trend: 14-day slope
        def rolling_slope(series, w=14):
            slopes = [np.nan] * len(series)
            arr = series.values
            for i in range(w, len(arr)):
                y = arr[i-w:i]
                if not np.any(np.isnan(y)):
                    slopes[i] = np.polyfit(np.arange(w), y, 1)[0]
            return slopes
        df_med["demand_trend_14"] = rolling_slope(df_med["quantity_sold"].shift(1))

        # ── External demand factors ───────────────────────
        if not demand_factors.empty:
            df_med = df_med.merge(
                demand_factors.rename(columns={"date": "sale_date"}),
                on="sale_date", how="left"
            )
            for col in ["is_holiday","is_promotion_active","is_ramadan",
                        "covid_period_flag","disease_trend_index"]:
                if col in df_med.columns:
                    df_med[col] = df_med[col].fillna(0)
        else:
            df_med["is_holiday"]           = 0
            df_med["is_promotion_active"]  = 0
            df_med["is_ramadan"]           = 0
            df_med["covid_period_flag"]    = 0
            df_med["disease_trend_index"]  = 0.5

        frames.append(df_med)

    if skipped:
        print(f"  Skipped {skipped} medicines with < {MIN_HISTORY} days of history")

    result = pd.concat(frames, ignore_index=True)
    print(f"Feature matrix: {len(result):,} rows × {len(result.columns)} columns "
          f"covering {result['generic_name'].nunique()} medicines\n")
    return result


FEATURE_COLS = [
    "medicine_id", "class_id",
    "day_of_week", "month", "day_of_month", "week_of_year", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "lag_1","lag_2","lag_3","lag_7","lag_14","lag_21","lag_28","lag_60","lag_90",
    "roll_mean_7",  "roll_std_7",  "roll_max_7",  "roll_min_7",
    "roll_mean_14", "roll_std_14", "roll_max_14", "roll_min_14",
    "roll_mean_30", "roll_std_30", "roll_max_30", "roll_min_30",
    "roll_mean_60", "roll_std_60",
    "roll_mean_90",
    "demand_trend_14",
    "is_holiday","is_promotion_active","is_ramadan",
    "covid_period_flag","disease_trend_index","is_winter",
]


# ─────────────────────────────────────────────────────────
# TRAIN / EVALUATE
# ─────────────────────────────────────────────────────────
def time_split(df: pd.DataFrame):
    max_date  = df["sale_date"].max()
    test_start = max_date - pd.Timedelta(days=TEST_DAYS - 1)
    val_start  = test_start - pd.Timedelta(days=VAL_DAYS)

    train = df[df["sale_date"] <  val_start]
    val   = df[(df["sale_date"] >= val_start) & (df["sale_date"] < test_start)]
    test  = df[df["sale_date"] >= test_start]

    print(f"Time split:")
    print(f"  Train : {train['sale_date'].min().date()} → {train['sale_date'].max().date()}  ({len(train):,} rows)")
    print(f"  Val   : {val['sale_date'].min().date()}   → {val['sale_date'].max().date()}    ({len(val):,} rows)")
    print(f"  Test  : {test['sale_date'].min().date()}  → {test['sale_date'].max().date()}   ({len(test):,} rows)\n")
    return train, val, test


def train_model(train: pd.DataFrame, val: pd.DataFrame):
    available   = [c for c in FEATURE_COLS if c in train.columns]
    train_clean = train[available + ["quantity_sold"]].dropna()
    val_clean   = val[available   + ["quantity_sold"]].dropna()

    X_train, y_train = train_clean[available], train_clean["quantity_sold"]
    X_val,   y_val   = val_clean[available],   val_clean["quantity_sold"]

    print(f"Training LightGBM global model")
    print(f"  Training rows : {len(X_train):,}")
    print(f"  Features      : {len(available)}")
    print(f"  Val rows      : {len(X_val):,}")
    print(f"  Early stopping: 50 rounds\n")

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=available)
    lgb_val   = lgb.Dataset(X_val,   label=y_val,   feature_name=available,
                            reference=lgb_train)

    model = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"\n✓ Best iteration: {model.best_iteration}")
    return model, available


def evaluate_model(model, test: pd.DataFrame, feature_cols: list):
    available  = [c for c in feature_cols if c in test.columns]
    test_clean = test[available + ["quantity_sold","generic_name"]].dropna()
    preds      = np.maximum(model.predict(test_clean[available]), 0)

    overall_mape = mean_absolute_percentage_error(test_clean["quantity_sold"], preds) * 100
    overall_mae  = mean_absolute_error(test_clean["quantity_sold"], preds)

    print(f"\n{'='*55}")
    print(f"  TEST SET PERFORMANCE  (last {TEST_DAYS} days)")
    print(f"  MAPE : {overall_mape:.2f}%")
    print(f"  MAE  : {overall_mae:.2f} units / day")
    print(f"{'='*55}\n")

    test_clean = test_clean.copy()
    test_clean["pred"] = preds
    per_med = test_clean.groupby("generic_name").apply(
        lambda g: pd.Series({
            "mape": mean_absolute_percentage_error(g["quantity_sold"], g["pred"]) * 100,
            "mae" : mean_absolute_error(g["quantity_sold"], g["pred"]),
        })
    ).round(2)

    print("Per-medicine MAPE (top 15 highest error):")
    print(per_med.sort_values("mape", ascending=False).head(15).to_string())
    return overall_mape, per_med


# ─────────────────────────────────────────────────────────
# FORECAST GENERATION
# ─────────────────────────────────────────────────────────
def generate_forecasts(
    model,
    feature_cols: list,
    df_all: pd.DataFrame,
    demand_factors: pd.DataFrame,
    per_med_mape: pd.DataFrame,
) -> list:

    MAX_HORIZON = max(HORIZONS)
    today       = datetime.now().date()
    records     = []

    print(f"\nGenerating forecasts for {df_all['generic_name'].nunique()} medicines …")

    for med_name in df_all["generic_name"].unique():
        df_med   = df_all[df_all["generic_name"] == med_name].sort_values("sale_date")
        if len(df_med) < MIN_HISTORY:
            continue

        med_id   = int(df_med["medicine_id"].iloc[0])
        class_id = int(df_med["class_id"].iloc[0])
        sku      = str(df_med["sku"].iloc[0])
        mape     = float(per_med_mape.loc[med_name, "mape"]) \
                   if med_name in per_med_mape.index else 15.0

        history      = list(df_med["quantity_sold"].tail(MAX_HORIZON + 90))
        future_preds = []

        for day_ahead in range(1, MAX_HORIZON + 1):
            fd  = pd.Timestamp(today) + pd.Timedelta(days=day_ahead)
            dow = fd.dayofweek
            mon = fd.month

            row = {
                "medicine_id":    med_id,
                "class_id":       class_id,
                "day_of_week":    dow,
                "month":          mon,
                "day_of_month":   fd.day,
                "week_of_year":   fd.isocalendar().week,
                "quarter":        (mon - 1) // 3 + 1,
                "is_weekend":     int(dow >= 4),
                "is_month_start": int(fd.day <= 3),
                "is_month_end":   int(fd.day >= 28),
                "month_sin":      np.sin(2 * np.pi * mon / 12),
                "month_cos":      np.cos(2 * np.pi * mon / 12),
                "dow_sin":        np.sin(2 * np.pi * dow / 7),
                "dow_cos":        np.cos(2 * np.pi * dow / 7),
                "is_winter":      int(mon in [1,2,3,10,11,12]),
                "demand_trend_14": 0.0,
            }

            h = history + future_preds
            for lag in [1,2,3,7,14,21,28,60,90]:
                row[f"lag_{lag}"] = h[-lag] if len(h) >= lag \
                                    else (np.mean(h[-5:]) if h else 1.0)

            for window in [7,14,30,60,90]:
                wv = h[-window:] if len(h) >= window else h
                row[f"roll_mean_{window}"] = float(np.mean(wv))  if wv else 1.0
                row[f"roll_std_{window}"]  = float(np.std(wv))   if len(wv)>1 else 0.0
                row[f"roll_max_{window}"]  = float(np.max(wv))   if wv else 1.0
                row[f"roll_min_{window}"]  = float(np.min(wv))   if wv else 0.0

            # External factors for future date
            if not demand_factors.empty:
                frow = demand_factors[demand_factors["date"].dt.date == fd.date()]
                row["is_holiday"]          = int(frow["is_holiday"].values[0])          if len(frow) else 0
                row["is_promotion_active"] = int(frow["is_promotion_active"].values[0]) if len(frow) else 0
                row["is_ramadan"]          = int(frow["is_ramadan"].values[0])           if len(frow) else 0
                row["covid_period_flag"]   = int(frow["covid_period_flag"].values[0])   if len(frow) else 0
                row["disease_trend_index"] = float(frow["disease_trend_index"].values[0]) if len(frow) else 0.5
            else:
                row.update({"is_holiday":0,"is_promotion_active":0,"is_ramadan":0,
                            "covid_period_flag":0,"disease_trend_index":0.5})

            avail = [c for c in feature_cols if c in row]
            X    = pd.DataFrame([row])[avail]
            pred = float(np.maximum(model.predict(X)[0], 0))
            future_preds.append(pred)

        for horizon in HORIZONS:
            hp    = future_preds[:horizon]
            total = sum(hp)
            records.append({
                "sku":              sku,
                "forecast_date":    today,
                "horizon_days":     horizon,
                "predicted_demand": round(total, 2),
                "confidence_lower": round(max(0.0, total * (1 - CONFIDENCE_PCT)), 2),
                "confidence_upper": round(total * (1 + CONFIDENCE_PCT), 2),
                "mape_score":       round(mape, 2),
                "model_version":    "LightGBM_v2.1",
                "generated_at":     datetime.now(),
            })

    print(f"✓ Generated {len(records)} forecast records "
          f"({len(records)//len(HORIZONS)} medicines × {len(HORIZONS)} horizons)")
    return records


# ─────────────────────────────────────────────────────────
# SAVE TO DATABASE
# ─────────────────────────────────────────────────────────
def save_forecasts(conn, records: list):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM demand_forecasts
            WHERE forecast_date = CURRENT_DATE
              AND model_version = 'LightGBM_v2.1'
        """)
        cursor.executemany("""
            INSERT INTO demand_forecasts (
                sku, forecast_date, horizon_days,
                predicted_demand, confidence_lower, confidence_upper,
                mape_score, model_version, generated_at
            ) VALUES (
                %(sku)s, %(forecast_date)s, %(horizon_days)s,
                %(predicted_demand)s, %(confidence_lower)s, %(confidence_upper)s,
                %(mape_score)s, %(model_version)s, %(generated_at)s
            )
        """, records)
        conn.commit()
        print(f"✓ Saved {len(records)} records to demand_forecasts")
    except Exception as e:
        conn.rollback()
        print(f"❌ Could not save forecasts: {e}")
    finally:
        cursor.close()


# ─────────────────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────────────────
def print_feature_importance(model, feature_cols):
    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print(f"\n{'='*50}")
    print("  TOP 20 FEATURES BY GAIN")
    print(f"{'='*50}")
    print(imp.head(20).to_string(index=False))


def print_summary(records, stock_df, df_all):
    df = pd.DataFrame(records)
    df30 = df[df["horizon_days"] == 30].copy()

    # Attach generic_name via sku
    sku_to_name = df_all.groupby("sku")["generic_name"].first().to_dict()
    df30["generic_name"] = df30["sku"].map(sku_to_name)

    if stock_df is not None and not stock_df.empty:
        df30 = df30.merge(stock_df, on="sku", how="left")

    print(f"\n{'='*72}")
    print("  30-DAY FORECAST SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Medicine':<30} {'Forecast':>10} {'Stock':>8} {'MAPE':>6}  Action")
    print(f"  {'-'*68}")

    urgent = reorder = monitor = ok = no_stock = 0
    for _, r in df30.sort_values("predicted_demand", ascending=False).iterrows():
        pred  = r["predicted_demand"]
        stock = r.get("current_stock", None)
        name  = str(r.get("generic_name", r["sku"]))[:28]

        if stock is None or (isinstance(stock, float) and np.isnan(stock)):
            action = "  — no stock data"
            no_stock += 1
        elif stock < pred * 0.5: action = "  🔴 URGENT ORDER";  urgent  += 1
        elif stock < pred:        action = "  🟠 REORDER";       reorder += 1
        elif stock < r["confidence_upper"]: action = "  🟡 MONITOR"; monitor += 1
        else:                     action = "  🟢 OK";             ok      += 1

        stock_str = f"{int(stock):>8}" if stock is not None and not (isinstance(stock,float) and np.isnan(stock)) else "       —"
        print(f"  {name:<30} {pred:>10.0f} {stock_str} {r['mape_score']:>5.1f}%{action}")

    print(f"\n  Urgent: {urgent}  |  Reorder: {reorder}  |  Monitor: {monitor}  "
          f"|  OK: {ok}  |  No stock data: {no_stock}")
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  MediBot AI Forecasting Engine v2.1")
    print("  LightGBM Global Model — all medicines")
    print("="*55 + "\n")

    t0   = datetime.now()
    conn = connect()

    # ── 1. Load all data ──────────────────────────────────
    sales_df       = load_sales(conn)
    demand_factors = load_demand_factors(conn)
    stock_df       = load_current_stock(conn)
    policies_df    = load_inventory_policies(conn)
    _suppliers_df  = load_suppliers(conn)

    if sales_df.empty:
        print("❌ No sales data found. Make sure sales_transactions is populated.")
        conn.close(); return

    # ── 2. Feature engineering ────────────────────────────
    df_features = build_features(sales_df, demand_factors)
    if df_features.empty:
        print("❌ Feature engineering produced no data. Check MIN_HISTORY setting.")
        conn.close(); return

    # ── 3. Time split ─────────────────────────────────────
    train, val, test = time_split(df_features)

    # ── 4. Train ──────────────────────────────────────────
    model, feature_cols = train_model(train, val)

    # ── 5. Evaluate ───────────────────────────────────────
    overall_mape, per_med_mape = evaluate_model(model, test, feature_cols)

    # ── 6. Feature importance ─────────────────────────────
    print_feature_importance(model, feature_cols)

    # ── 7. Save model ─────────────────────────────────────
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}")

    # ── 8. Generate forecasts ─────────────────────────────
    records = generate_forecasts(model, feature_cols, df_features,
                                 demand_factors, per_med_mape)

    # ── 9. Save forecasts to DB ───────────────────────────
    save_forecasts(conn, records)

    # ── 10. Summary ───────────────────────────────────────
    print_summary(records, stock_df, df_features)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Runtime     : {elapsed:.1f} seconds")
    print(f"  Overall MAPE: {overall_mape:.2f}%")
    print(f"  Model file  : {MODEL_PATH}")
    print(f"  Forecasts   : demand_forecasts table\n")

    conn.close()


if __name__ == "__main__":
    main()
