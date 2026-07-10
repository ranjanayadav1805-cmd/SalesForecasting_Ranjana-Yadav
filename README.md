# 📈 End-to-End Sales Forecasting and Demand Intelligence System

> A complete Machine Learning and Business Intelligence solution for retail sales forecasting, anomaly detection, product demand segmentation, and interactive decision support.


---


## 🚀 Live Demo

*Streamlit App:  https://salesforecastingranjana-yadav-zsxhk83ubl9lmfur6gbu9o.streamlit.app/


---


# 📖 Project Description

Retail businesses generate large volumes of transactional data every day. Making informed decisions about inventory, product demand, and future sales requires accurate forecasting and business analytics.

This project develops an *End-to-End Sales Forecasting and Demand Intelligence System* using the Superstore Sales dataset. The solution combines Exploratory Data Analysis (EDA), Time Series Forecasting, Anomaly Detection, Product Segmentation, and an Interactive Streamlit Dashboard to support data-driven business decisions.

---

# 🎯 Project Objectives

- Analyze historical sales performance
- Forecast future sales using Machine Learning models
- Detect abnormal sales behaviour
- Segment products according to demand
- Build an interactive Streamlit dashboard
- Generate actionable business recommendations

---

# 📂 Dataset Information

*Dataset:* Superstore Sales Dataset (train.csv)

### Dataset Features

- Order ID
- Order Date
- Ship Date
- Customer ID
- Customer Name
- Segment
- Country
- City
- State
- Region
- Product ID
- Category
- Sub-Category
- Product Name
- Sales
- Quantity
- Discount
- Profit

The dataset contains historical retail transactions used for forecasting and business analysis.

---

# 🔬 Project Workflow

## Task 1 — Data Preparation & Exploratory Data Analysis

- Data Cleaning
- Missing Value Analysis
- Duplicate Removal
- Data Type Conversion
- Exploratory Data Analysis
- Sales Trend Analysis

---

## Task 2 — Time Series Analysis

- Monthly Sales Aggregation
- Trend Analysis
- Seasonal Decomposition
- Stationarity Testing (ADF Test)

---

## Task 3 — Sales Forecasting

Implemented three forecasting models:

- SARIMA
- Facebook Prophet
- XGBoost Regressor

### Model Evaluation

Evaluation Metrics

- MAE
- RMSE
- MAPE

The best-performing model was selected based on forecasting accuracy.

---

## Task 4 — Category & Region Forecasting

Sales forecasts were generated for:

- Furniture
- Office Supplies
- Technology

Regional Forecasting

- East
- West

Forecast horizon:

- Next 3 Months

---

## Task 5 — Sales Anomaly Detection

Two anomaly detection techniques were implemented:

### Isolation Forest

Detects unusual sales observations using an unsupervised learning algorithm.

### Z-Score Method

Identifies statistical outliers based on standard deviation.

Detected anomalies help identify:

- Promotional events
- Seasonal spikes
- Unexpected sales drops

---

## Task 6 — Product Demand Segmentation

Products were segmented using *K-Means Clustering*.

Segmentation helps businesses:

- Identify high-demand products
- Optimize inventory
- Improve stocking strategies

Cluster visualization was created for better interpretation.

---

## Task 7 — Interactive Dashboard (Streamlit)

The project includes a multi-page interactive dashboard.

### Dashboard Pages

### 📊 Sales Overview

- Sales KPIs
- Yearly Sales
- Monthly Trend
- Filters

### 📈 Forecast Explorer

- Forecast by Category
- Forecast by Region
- Performance Metrics

### 🚨 Anomaly Report

- Isolation Forest Results
- Z-Score Results
- Anomaly Table

### 📦 Product Demand Segments

- Cluster Visualization
- Product Categories
- Demand Insights

---

# 🛠️ Technologies Used

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Statsmodels
- Prophet
- XGBoost
- Scikit-learn
- Streamlit

---

# 📊 Business Insights

The analysis revealed:

- Technology generated the highest revenue.
- Sales exhibited clear seasonal trends.
- High-demand products require optimized inventory planning.
- Anomaly detection highlighted unusual sales events.
- Forecasting enables proactive inventory management.

---

# 💼 Business Recommendations

- Increase inventory before high-demand periods.
- Monitor anomalies for operational risks.
- Use demand segmentation for stock optimization.
- Continuously retrain forecasting models with new data.
- Use dashboard insights for strategic planning.

---

# ▶️ Installation

Clone the repository

bash
git clone https://github.com/ranjanayadav1805-cmd/SalesForecasting_Ranjana-Yadav


Install dependencies

bash
pip install -r requirements.txt


Run the dashboard

bash
streamlit run app.py


