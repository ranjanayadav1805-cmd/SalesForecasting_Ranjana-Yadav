import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.title("📊 End-to-End Sales Forecasting & Demand Intelligence System")

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("train.csv")

# Convert Date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

sales_df = df.copy()

# -------------------------
# Sidebar
# -------------------------
page = st.sidebar.selectbox(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# ==================================================
# PAGE 1
# ==================================================
if page == "Sales Overview":

    st.header("📈 Sales Overview Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"${df['Sales'].sum():,.2f}")
    col2.metric("Average Sales", f"${df['Sales'].mean():,.2f}")
    col3.metric("Total Orders", df["Order ID"].nunique())

    # Sales by Year
    yearly = (
        df.groupby(df["Order Date"].dt.year)["Sales"]
        .sum()
        .reset_index()
    )

    yearly.columns = ["Year", "Sales"]

    fig = px.bar(
        yearly,
        x="Year",
        y="Sales",
        title="Total Sales by Year"
    )

    st.plotly_chart(fig, use_container_width=True)
    # Convert date column
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

        # Monthly Sales Trend

monthly = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )
monthly.columns = ["ds", "y"]
monthly = monthly.rename(
        columns={
            "Order Date": "ds",
            "Sales": "y"
        }
    )

fig2 = px.line(
    monthly,
    x="ds",
    y="y",
    title="Monthly Sales Trend"
)

st.plotly_chart(fig2, use_container_width=True, key="monthly_sales_trend")
    # Region Filter
st.subheader("Sales by Region")

region = st.selectbox(
        "Select Region",
        df["Region"].unique()
    )
filtered = df[df["Region"] == region]

region_sales = (
        filtered.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

fig3 = px.bar(
        region_sales,
        x="Category",
        y="Sales",
        color="Category",
        title="Sales by Category"
    )
st.plotly_chart(fig3, use_container_width=True)

# ==================================================
# PAGE 2 - FORECAST EXPLORER
# ==================================================

if page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    option = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    if option == "Category":
        selected = st.selectbox(
            "Select Category",
            sales_df["Category"].unique(),
            key="region_select_1"
        )

        data = sales_df[sales_df["Category"] == selected]

    else:
        selected = st.selectbox(
            "Select Region",
            sales_df["Region"].unique(),
             key="region_select_2"
        )

        data = sales_df[sales_df["Region"] == selected]

    monthly = (
        data.groupby("Order Date")["Sales"]
        .sum()
        .reset_index()
    )

    monthly.columns = ["ds", "y"]

    # Train/Test Split
    train = monthly.iloc[:-12]
    test = monthly.iloc[-12:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=12, freq="ME")
    forecast = model.predict(future)

    # Forecast Graph
    fig = model.plot(forecast)
    st.pyplot(fig)

    # Calculate MAE & RMSE
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    pred = forecast[["ds", "yhat"]].tail(12)

    mae = mean_absolute_error(test["y"], pred["yhat"])
    rmse = np.sqrt(mean_squared_error(test["y"], pred["yhat"]))

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("MAE", round(mae, 2))
    col2.metric("RMSE", round(rmse, 2))

    # Forecast Horizon
    horizon = st.slider(
        "Forecast Horizon (Months)",
        1, 3, 3
    )

    future2 = model.make_future_dataframe(periods=horizon, freq="ME")
    forecast2 = model.predict(future2)

    st.subheader("Upcoming Forecast")

    st.dataframe(
        forecast2[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
    )

# ==================================================
# PAGE 3 - ANOMALY REPORT
# ==================================================

if page == "Anomaly Report":

    st.header("🚨 Anomaly Report")

    anomaly_data = sales_df.groupby("Order Date")["Sales"].sum().reset_index()

    model = IsolationForest(contamination=0.05, random_state=42)

    anomaly_data["Anomaly"] = model.fit_predict(anomaly_data[["Sales"]])

    anomalies = anomaly_data[anomaly_data["Anomaly"] == -1]

    st.subheader("Detected Anomalies")

    st.dataframe(anomalies)

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        anomaly_data["Order Date"],
        anomaly_data["Sales"],
        label="Sales"
    )

    ax.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        s=70,
        label="Anomaly"
    )

    ax.set_title("Sales Anomaly Detection")

    ax.legend()

    st.pyplot(fig)

# ==================================================
# PAGE 4 - PRODUCT DEMAND SEGMENTS
# ==================================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segments")

    product = (
    sales_df.groupby("Sub-Category")["Sales"]
    .sum()
    .reset_index()
)
    scaler = StandardScaler()

    X = scaler.fit_transform(product[["Sales",]])

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    product["Cluster"] = model.fit_predict(X)

st.subheader("Cluster Table")
product = sales_df.groupby("Sub-Category")["Sales"].sum().reset_index()
st.dataframe(product)


fig, ax = plt.subplots(figsize=(12,6))

ax.bar(
    product["Sub-Category"],
    product["Sales"],
    color="skyblue"
)

plt.xticks(rotation=90)

ax.set_xlabel("Sub Category")
ax.set_ylabel("Sales")
ax.set_title("Product Demand Segments")

st.pyplot(fig)