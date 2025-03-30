Medical Insurance Cost Prediction and Fairness Analysis






This project aims to predict individual medical insurance charges using machine learning, while ensuring fairness and transparency in the process. The project explores patterns in demographic and health-related factors (like age, BMI, smoking status) and uses clustering and explainability tools to gain actionable insights.

- Project Goals

Predict medical insurance costs using ML models

Analyze feature influence on pricing

Cluster patients based on health risk

Ensure fair pricing across demographic groups

- Techniques Used
  
EDA & Visualization: Seaborn, Matplotlib

Feature Engineering: BMI category, age buckets, risk factor

Clustering: K-Means (3 clusters based on age, BMI, smoking status)

Dimensionality Reduction: PCA for visualization

Planned Modeling: XGBoost, Random Forest

Explainability (Planned): SHAP

Fairness Analysis (Planned): Bias and group fairness checks

📈 Preliminary Insights
Smokers pay significantly higher charges

BMI and Age positively correlate with insurance cost

Three Risk Segments Identified using KMeans:

High-Risk Smokers (~$32,000)

Average-Risk Non-Smokers (~$11,600)

Low-Risk Healthy Individuals (~$5,000)

✅ Next Steps
Train predictive models (XGBoost, Random Forest)

Apply SHAP for model interpretability

Conduct fairness testing across demographic groups

Finalize evaluation metrics (R², RMSE)

Prepare model dashboard or notebook walkthrough

📊 Dataset Source
Kaggle - Medical Insurance Cost Dataset

