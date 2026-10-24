import psycopg2
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Connect to database ──────────────────────────────────────────
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="PHARMACY_DB",
    user="postgres",
    password="407090ST@"
)

print("Connected to database...")
print("Loading sales history...")

# ─── Load sales data ──────────────────────────────────────────────
query = """
    SELECT
        m.generic_name,
        m.sku::text,
        DATE(st.dispensed_at) AS sale_date,
        SUM(st.quantity_dispensed) AS daily_qty
    FROM sales_transactions st
    JOIN medicines m ON m.sku = st.sku
    GROUP BY m.generic_name, m.sku, DATE(st.dispensed_at)
    ORDER BY m.generic_name, sale_date
"""
df = pd.read_sql(query, conn)
print(f"Loaded {len(df)} daily sales records across {df['generic_name'].nunique()} medicines")

# ─── Feature engineering ──────────────────────────────────────────
def build_features(df_med):
    df_med = df_med.copy().sort_values('sale_date')
    df_med['sale_date'] = pd.to_datetime(df_med['sale_date'])
    df_med['day_of_week']  = df_med['sale_date'].dt.dayofweek
    df_med['month']        = df_med['sale_date'].dt.month
    df_med['day_of_month'] = df_med['sale_date'].dt.day
    df_med['is_weekend']   = (df_med['day_of_week'] >= 5).astype(int)
    df_med['is_winter']    = df_med['month'].isin([11,12,1,2,3]).astype(int)
    df_med['lag_1']  = df_med['daily_qty'].shift(1)
    df_med['lag_7']  = df_med['daily_qty'].shift(7)
    df_med['lag_14'] = df_med['daily_qty'].shift(14)
    df_med['rolling_7']  = df_med['daily_qty'].shift(1).rolling(7).mean()
    df_med['rolling_14'] = df_med['daily_qty'].shift(1).rolling(14).mean()
    df_med['rolling_30'] = df_med['daily_qty'].shift(1).rolling(30).mean()
    return df_med.dropna()

features = ['day_of_week','month','day_of_month','is_weekend',
            'is_winter','lag_1','lag_7','lag_14',
            'rolling_7','rolling_14','rolling_30']

# ─── Train model per medicine ─────────────────────────────────────
print("\nTraining forecasting models...")
print("-" * 50)

results = []
cursor = conn.cursor()

for medicine_name in df['generic_name'].unique():
    df_med = df[df['generic_name'] == medicine_name].copy()
    sku = df_med['sku'].iloc[0]
    
    df_feat = build_features(df_med)
    
    if len(df_feat) < 30:
        print(f"  Skipping {medicine_name} — insufficient data")
        continue
    
    X = df_feat[features]
    y = df_feat['daily_qty']
    
    # Train/test split (last 30 days = test)
    split = len(df_feat) - 30
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print(f"  {medicine_name:<25} MAPE: {mape:.1f}%  |  Avg daily demand: {y.mean():.1f} units")
    
    # Generate forecasts for next 14 and 30 days
    last_row = df_feat.iloc[-1].copy()
    
    for horizon in [7, 14, 30]:
        future_preds = []
        rolling_window = list(df_feat['daily_qty'].tail(30))
        
        for day_ahead in range(horizon):
            future_date = pd.Timestamp(df_feat['sale_date'].max()) + timedelta(days=day_ahead+1)
            
            lag_1  = rolling_window[-1]
            lag_7  = rolling_window[-7]  if len(rolling_window) >= 7  else np.mean(rolling_window)
            lag_14 = rolling_window[-14] if len(rolling_window) >= 14 else np.mean(rolling_window)
            
            row = pd.DataFrame([{
                'day_of_week':  future_date.dayofweek,
                'month':        future_date.month,
                'day_of_month': future_date.day,
                'is_weekend':   int(future_date.dayofweek >= 5),
                'is_winter':    int(future_date.month in [11,12,1,2,3]),
                'lag_1':        lag_1,
                'lag_7':        lag_7,
                'lag_14':       lag_14,
                'rolling_7':    np.mean(rolling_window[-7:]),
                'rolling_14':   np.mean(rolling_window[-14:]),
                'rolling_30':   np.mean(rolling_window[-30:]),
            }])
            
            pred = max(0, model.predict(row)[0])
            future_preds.append(pred)
            rolling_window.append(pred)
        
        total_predicted = sum(future_preds)
        avg_daily = np.mean(future_preds)
        confidence_lower = max(0, total_predicted * 0.80)
        confidence_upper = total_predicted * 1.20
        
        # Save forecast to database
        # Save forecast to database
        cursor.execute("""
            INSERT INTO demand_forecasts (
                sku, forecast_date, horizon_days,
                predicted_demand, confidence_lower, confidence_upper,
                mape_score, model_version, generated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            sku,
            datetime.now().date(),
            horizon,
            float(round(total_predicted, 2)),
            float(round(confidence_lower, 2)),
            float(round(confidence_upper, 2)),
            float(round(mape, 2)),
            'GBR_v1.0',
            datetime.now()
        ))

        
        results.append({
            'medicine': medicine_name,
            'horizon_days': horizon,
            'predicted_demand': round(total_predicted, 1),
            'confidence_lower': round(confidence_lower, 1),
            'confidence_upper': round(confidence_upper, 1),
            'mape': round(mape, 1)
        })

conn.commit()

# ─── Print forecast summary ───────────────────────────────────────
print("\n" + "=" * 60)
print("DEMAND FORECAST SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results)
for medicine in results_df['medicine'].unique():
    med_results = results_df[results_df['medicine'] == medicine]
    print(f"\n{medicine}")
    for _, row in med_results.iterrows():
        print(f"  Next {row['horizon_days']:2d} days: {row['predicted_demand']:6.0f} units "
              f"(range: {row['confidence_lower']:.0f} - {row['confidence_upper']:.0f}) "
              f"MAPE: {row['mape']:.1f}%")

print("\n✓ All forecasts saved to demand_forecasts table")
print("✓ AI forecasting model complete!")

cursor.close()
conn.close()

