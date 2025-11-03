import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import warnings

st.set_page_config(layout="centered", page_title="RFM Predictor")
st.title("🤖 RFM Segment Predictor (K-Means)")
st.write("Enter a customer's Recency, Frequency, and Monetary values to predict their K-Means segment.")
st.info("This model was trained on the Olist dataset (using K=11) to identify natural customer behavior patterns.")

DATA_DIR = Path("data")

@st.cache_data
def train_prediction_model():
    warnings.filterwarnings('ignore')
    try:
        df_orders = pd.read_csv(DATA_DIR / 'olist_orders_dataset.csv')
        df_payments = pd.read_csv(DATA_DIR / 'olist_order_payments_dataset.csv')
        df_customers = pd.read_csv(DATA_DIR / 'olist_customers_dataset.csv')
    except FileNotFoundError:
        st.error(f"Error: CSV files not found. Please make sure the 'data' folder is in the same directory as app.py.")
        st.stop()
        
    df_orders_delivered = df_orders[df_orders['order_status'] == 'delivered'].copy()
    df_payments_agg = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    df_merged = df_orders_delivered.merge(df_customers, on='customer_id')
    df_merged = df_merged.merge(df_payments_agg, on='order_id')
    df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
    today = df_merged['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm_df = df_merged.groupby('customer_unique_id').agg(
        last_purchase_date=('order_purchase_timestamp', 'max'),
        frequency=('order_id', 'nunique'),
        monetary=('payment_value', 'sum')
    ).reset_index() 
    rfm_df['recency'] = (today - rfm_df['last_purchase_date']).dt.days
    rfm_df = rfm_df[['customer_unique_id', 'recency', 'frequency', 'monetary']]

    rfm_raw = rfm_df[['recency', 'frequency', 'monetary']]
    rfm_log = np.log1p(rfm_raw)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    k = 11
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(rfm_scaled)
    
    scaled_mean = scaler.mean_
    scaled_std = scaler.scale_
    centroids_scaled = kmeans.cluster_centers_

    def name_cluster(centroid, scaled_mean, scaled_std):
        r_score, f_score, m_score = centroid
        HIGH = 0.5
        LOW = -0.5
        
        if r_score < LOW and f_score > HIGH and m_score > HIGH:
            return 'K-Champions'
        elif r_score > HIGH and f_score > HIGH and m_score > HIGH:
            return 'K-Can’t Lose Them'
        elif r_score > LOW and f_score > HIGH and m_score > HIGH:
            return 'K-At Risk (High Value)'
        elif r_score < LOW and f_score > LOW and m_score > LOW:
            return 'K-Loyal'
        elif r_score < LOW and f_score < LOW and m_score < LOW:
            return 'K-New/Promising'
        elif r_score > HIGH and f_score < LOW and m_score < LOW:
            return 'K-Lost'
        elif r_score > LOW and f_score < LOW and m_score < LOW:
            return 'K-About to Sleep'
        elif r_score > 0:
            return 'K-Lapsing'
        else:
            return 'K-Active (Mid-Value)'

    cluster_map = {}
    for i, centroid in enumerate(centroids_scaled):
        cluster_map[i] = name_cluster(centroid, scaled_mean, scaled_std)
    return kmeans, scaler, cluster_map

SEGMENT_DEFINITIONS = {
    "K-Champions": "Recent, very frequent, high spenders. Your best customers.",
    "K-Can’t Lose Them": "Used to be frequent & high spenders, but haven't returned in a long time.",
    "K-At Risk (High Value)": "Similar to 'Can't Lose Them', but more recent.",
    "K-Loyal": "Loyal customers who are recent, with above-average frequency & monetary.",
    "K-New/Promising": "New customers with low frequency & monetary. Need nurturing.",
    "K-Lost": "Haven't bought in a very long time, low frequency & monetary.",
    "K-About to Sleep": "Haven't bought in a while, low frequency & monetary.",
    "K-Lapsing": "Fading customers (Recency > 0, meaning less recent).",
    "K-Active (Mid-Value)": "Active customers (Recency < 0, meaning more recent) with mid-range value."
}

with st.spinner("Training K-Means model on historical data..."):
    kmeans_model, scaler, cluster_map = train_prediction_model()

st.sidebar.header("Segment Definitions")
st.sidebar.write("These names are based on the K-Means cluster centroid analysis:")
for segment, description in SEGMENT_DEFINITIONS.items():
    st.sidebar.subheader(f"Segment: {segment}")
    st.sidebar.write(description)

st.header("🎛️ Customer Segment Simulator")

with st.container(border=True):
    with st.form(key="prediction_form"):
        st.write("**Enter customer values:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            recency_input = st.number_input(
                "Recency (Days since last purchase)", 
                min_value=0, value=30
            )
        with col2:
            frequency_input = st.number_input(
                "Frequency (Total purchases)", 
                min_value=1, value=1
            )
        with col3:
            monetary_input = st.number_input(
                "Monetary (Total spend R$)", 
                min_value=0.0, value=150.0, format="%.2f"
            )
        
        submit_button = st.form_submit_button(label="Predict Segment", type="primary")

if submit_button:
    user_data_raw = pd.DataFrame({
        'recency': [recency_input],
        'frequency': [frequency_input],
        'monetary': [monetary_input]
    })
    
    user_data_log = np.log1p(user_data_raw)
    
    user_data_scaled = scaler.transform(user_data_log)
    
    predicted_cluster = kmeans_model.predict(user_data_scaled)[0]
    
    predicted_segment = cluster_map[predicted_cluster]
    
    st.header("📈 Prediction Result")
    with st.container(border=True):
        st.metric(label="Predicted Customer Segment (K-Means)", value=predicted_segment)
        
        # Show definition right with the result
        st.info(f"**What does this mean?**\n\n{SEGMENT_DEFINITIONS.get(predicted_segment, 'No definition available.')}")
        
        st.caption(f"This customer (R={recency_input}, F={frequency_input}, M={monetary_input}) was assigned to internal **Cluster ID: {predicted_cluster}**.")