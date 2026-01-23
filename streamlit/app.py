import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
load_dotenv()

# -----------------------------
# Database connection
# -----------------------------
def get_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],      
        port=os.environ["DB_PORT"],      
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

def load_df(query: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(query, conn)

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(
    page_title="EUR/USD Forecast",
    layout="wide"
)

st.title("EUR/USD Daily Forecast")

# -----------------------------
# 1. Latest forecast
# -----------------------------
latest_pred = load_df("""
    SELECT *
    FROM eur_usd_predictions
    ORDER BY feature_date DESC
    LIMIT 1
""")

st.subheader("Tomorrow's Forecast")

if latest_pred.empty:
    st.warning("No forecast available.")
else:
    row = latest_pred.iloc[0]

    direction_map = {
        1: "📈 Up",
        0: "📉 Down"
    }

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction Date", row["prediction_date"].strftime("%Y-%m-%d"))
    col2.metric("Direction", direction_map[row["predicted_direction"]])
    col3.metric("Model", f"{row['model_name']} (v{row['model_version']})")

# -----------------------------
# 2. Load historical data
# -----------------------------
prices = load_df("""
    SELECT 
        datetime,
        open,
        high,
        low,
        close
    FROM eur_usd_final
    ORDER BY datetime DESC;
""").sort_values("datetime")

predictions = load_df("""
    SELECT feature_date, prediction_date, predicted_direction
    FROM eur_usd_predictions
    ORDER BY feature_date DESC
""")

# -----------------------------
# 3. Actual price chart
# -----------------------------
st.subheader("📊 EUR/USD Actual Prices")

prices = prices.sort_values("datetime")

fig = go.Figure(
    data=[
        go.Candlestick(
            x=prices["datetime"],
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
        )
    ]
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="EUR/USD Price",
    xaxis_rangeslider_visible=False,
    height=500
)

st.plotly_chart(fig, width="stretch")
# -----------------------------
# 4. Predicted vs Actual Direction
# -----------------------------
st.subheader("Past Forecasts vs Reality")

# Actual direction:
# 1 → price went up
# 0 → price stayed same or went down
prices["actual_direction"] = (
    prices["close"].diff() > 0
).astype(int)

comparison = predictions.merge(
    prices,
    left_on="prediction_date",
    right_on="datetime",
    how="left"
)

comparison = comparison.dropna(subset=["actual_direction"])

comparison["prediction"] = comparison["predicted_direction"].map({
    1: "⬆️ Up",
    0: "⬇️ Down"
})

comparison["actual"] = comparison["actual_direction"].map({
    1: "⬆️ Up",
    0: "⬇️ Down"
})

comparison["correct"] = (
    comparison["predicted_direction"] == comparison["actual_direction"]
)

comparison["Result"] = comparison["correct"].map({
    True: "✅ Correct",
    False: "❌ Incorrect"
})


def highlight_result(row):
    if row["Result"] == "✅ Correct":
        return ["background-color: #d4edda"] * len(row)
    else:
        return ["background-color: #f8d7da"] * len(row)

# Display comparison table
display_cols = [
    "feature_date",
    "prediction_date",
    "prediction",
    "actual",
    "Result",
]

styled_df = (
    comparison[display_cols]
    .sort_values("feature_date", ascending=False)
    .style
    .apply(highlight_result, axis=1)
)

st.dataframe(styled_df, width="stretch")

# -----------------------------
# 5. Accuracy metric
# -----------------------------
if not comparison.empty:
    accuracy = comparison["correct"].mean() * 100
    st.metric("Directional Accuracy (recent)", f"{accuracy:.2f}%")
