Medical Insurance Cost Prediction & Fairness Analysis


Hi there! 👋 This is an ongoing project where I'm working to predict medical insurance charges based on a person's age, BMI, smoking status, gender, and region using machine learning. But I also want to go a step further — I aim to understand if the charges are fair across different groups and make the prediction process more transparent and explainable.

🎯 Project Motivation


Health insurance is getting more expensive and harder to understand. People often get charged differently based on their personal details, and it’s not always clear why. This project was born out of curiosity:

Can we build a model that predicts insurance costs accurately, and also make sure it’s fair to everyone?

By using clustering, visualization, and upcoming tools like SHAP and fairness metrics, I’m exploring not just prediction — but also insight and accountability.

📌 What’s Been Done So Far


✅ Loaded and cleaned the Medical Insurance Cost dataset
✅ Explored the data through visualizations (age, BMI, charges, etc.)
✅ Performed feature engineering (BMI category, age groups, risk factors)
✅ Used KMeans clustering to segment patients into 3 risk groups
✅ Applied PCA to visualize the clusters in 2D space

🔍 Clustering Insights


Using only age, BMI, and smoking status, KMeans discovered 3 natural groups:

Cluster Label	Key Traits	Avg. Charges


High-Risk Smokers	All smokers, higher BMI	~$32,050
Average-Risk Non-Smokers	Older, average BMI, non-smokers	~$11,624
Low-Risk Healthy Individuals	Young, healthy, non-smokers	~$5,065
This helps us understand how different behaviors and lifestyles affect cost, and it also opens up the idea of building cluster-specific models later.

🔧 What’s Next


🧠 Train models (starting with Random Forest, XGBoost)
📊 Use SHAP to explain which features drive predictions
⚖️ Perform fairness analysis (smoking, gender, BMI bias?)
📈 Compare metrics (RMSE, R², MAE)
📂 Possibly turn it into a dashboard or Streamlit app

🗃️ Dataset Info
Source: Kaggle – Medical Insurance Dataset

Size: 2,772 rows × 7 columns

Features: Age, Sex, BMI, Children, Smoker, Region, Charges
