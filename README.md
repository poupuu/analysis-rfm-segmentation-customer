# E-Commerce Customer Segmentation (RFM & K-Means)

## Summary
This project develops an unsupervised machine learning model (K-Means Clustering) to segment e-commerce customers based on their purchasing behavior (Recency, Frequency, Monetary). It then deploys this trained model into an interactive Streamlit application that can predict the customer segment for any new user-inputted RFM values.

## Problem
A "one-size-fits-all" marketing strategy is inefficient and costly. This project addresses two primary problems:
1.  **Inefficient Marketing Spend:** Treating all customers identically, (e.g., offering large discounts to "Champions" who would have purchased anyway).
2.  **Customer Churn & Lost Revenue:** Failing to identify and re-engage high-value "At Risk" or "Can’t Lose Them" customers before they are lost forever.

## Methodology
1.  **Data Loading & Preprocessing:** Loaded and merged three separate Olist datasets (orders, payments, customers). Filtered for 'delivered' orders only.
2.  **RFM Feature Engineering:** Calculated Recency (R), Frequency (F), and Monetary (M) values for every unique customer from their order history.
3.  **Data Transformation (Scaling):** Addressed high skew in the data (common in RFM) using a `log1p` transformation, then normalized the data using `StandardScaler` to prepare it for K-Means.
4.  **Modeling (K-Means Clustering):**
    * Initially analyzed data using a traditional Rule-Based (Quantile) method.
    * Trained a `KMeans` model (K=11) to discover natural, data-driven clusters, which proved more accurate than the rule-based method.
5.  **Cluster Analysis & Naming:** Analyzed the `centroids` (scaled R, F, M values) of each of the 11 clusters to give them descriptive, business-focused names (e.g., 'K-Champions', 'K-Lost').
6.  **Deployment:** Built an interactive Streamlit application (`app.py`) that serves the trained `KMeans` model and `scaler` to predict the segment for new, user-inputted data in real-time.

## Skills
1.  **Python:** Pandas, Numpy, Matplotlib/Seaborn
2.  **Data Preprocessing:** `pd.merge`, `groupby`, `.agg()`, `pd.to_datetime`
3.  **ML Preprocessing:** `StandardScaler` (Scaling), `np.log1p` (Log Transformation)
4.  **Modeling:** `KMeans` (Unsupervised Learning, Clustering) from `scikit-learn`
5.  **Deployment / Web App:** `Streamlit` (including `@st.cache_data` for performance)

## Results
1.  **Model:** A trained K-Means model that successfully identifies 9-11 distinct, actionable customer segments (e.g., 'K-Champions', 'K-Can’t Lose Them', 'K-New/Promising') based on their natural data patterns.
2.  **Key Insight:** The K-Means model proved that the traditional Rule-Based (Quantile) method was highly misleading for this dataset. The rule-based method incorrectly labeled thousands of 'one-time-purchase' customers as 'Champions', while K-Means correctly identified the *true* Champions segment as a much smaller, elite group.
3.  **Tool:** A fully functional and interactive Streamlit web application (`rfm_predictor_app.py`) that serves as a "Segment Predictor" tool, allowing any user to input R, F, & M values and get an instant segment prediction.

## Next Steps
1.  **Separate Training/Inference:** Package the `scaler` and `kmeans` model as `.pkl` files (pickling) to create a formal MLOps pipeline, separating the one-time training process from the lightweight prediction app.
2.  **Try Advanced Models:** Experiment with other clustering algorithms like **DBSCAN** or **Gaussian Mixture Models (GMM)** to see if they find more robust or different segment structures.
3.  **Richer Feature Engineering:** Incorporate more features beyond just RFM to create even richer customer personas, such as **product categories purchased**, **review scores**, or **payment types used**.
```eof
 
