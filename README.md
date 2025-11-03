# E-Commerce Customer Segmentation (RFM & K-Means)

## Summary
This project develops an unsupervised machine learning model (K-Means Clustering) to segment e-commerce customers based on their purchasing behavior (Recency, Frequency, Monetary). It then deploys this trained model into an interactive Streamlit application that can predict the customer segment for any new user-inputted RFM values.

## Problem
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
2.  **Data Preprocessing:** Merge, groupby, aggregate, datetime
3.  **ML Preprocessing:** Scaling, Log Transformation
4.  **Modeling:** Kmeans
5.  **Deployment / Web App:** Streamlit

## Results
1. The Rule-Based Method was Misleading. The simple "Rule-Based" method failed because 90% of customers only purchased once (Frequency=1). The model couldn't tell these identical customers apart and wrongly labeled thousands of them as "Champions," creating a segment of over 15,000.
2. K-Means Found the True Champions: The K-Means (Machine Learning) model analyzed the natural behavior and correctly identified the true "Champions" (customers who buy very frequently) as a tiny, elite group of only about 500 people.
3. A More Realistic Business View: The K-Means model shows a more realistic business picture: the largest segment by far is "K-Lost Customers" (around 20,000 people), a fact that was completely hidden by the flawed rule-based method.

## Next Steps
1.  **Separate Training/Inference:** Package the `scaler` and `kmeans` model as `.pkl` files (pickling) to create a formal MLOps pipeline, separating the one-time training process from the lightweight prediction app.
2.  **Try Advanced Models:** Experiment with other clustering algorithms like **DBSCAN** or **Gaussian Mixture Models (GMM)** to see if they find more robust or different segment structures.
3.  **Richer Feature Engineering:** Incorporate more features beyond just RFM to create even richer customer personas, such as **product categories purchased**, **review scores**, or **payment types used**.
