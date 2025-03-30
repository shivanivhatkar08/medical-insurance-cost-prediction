import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.model_selection import train_test_split

#Loading the dataset

med_ds = pd.read_csv(r"C:\Users\tvhat\OneDrive\Desktop\VS CODE\Python\Medical Insurance\medical_insurance.csv")
print(med_ds.head())

print(med_ds.info()) # this dataset has 2772 entries and 7 columns which are age, sex, bmi, children, smoker, region, charges
print(med_ds.dtypes)
print(med_ds.describe())
print(med_ds.isnull().sum())

#checking for duplicates
print("Number of duplicate records:", med_ds.duplicated().sum())
print("Duplicate records:", med_ds.duplicated())

med_ds = med_ds.drop_duplicates()
print("New dataset after removing duplicates:", med_ds.shape)

gen_count = med_ds['sex'].value_counts()
male_cnt = gen_count.get('male', 0)
female_cnt = gen_count.get('female',0)

total_reg = med_ds.groupby('region')['charges'].sum()


#Understanding the dataset 
print(f"""
      
Number of Males: {male_cnt}
Number of Females: {female_cnt}
Highest Age: {med_ds['age'].max()}
Lowest Age : {med_ds['age'].min()} 
Average Age: {med_ds['age'].mean(): .2f}

Average BMI: {med_ds['bmi'].mean():.2f}
Average Number of Children: {med_ds['children'].mean():.2f}

Highest Medical Insurance Charges: {med_ds['charges'].max():.2f}
Lowest Medical Insurance Charges: {med_ds['charges'].min():.2f}

Most Expensive Medical Insurance Region: {total_reg.idxmax() , f"(${total_reg.max():.2f})"}
Least Expensive Medical Insurance Region: {total_reg.idxmin(), f"(${total_reg.min():.2f})"}


""")


#Converting sex, smoker, region into categorical columns to numeric
med_ds['sex'] = med_ds['sex'].map({'male' : 1, 'female' : 0})
med_ds['smoker'] = med_ds['smoker'].map({'yes': 1, 'no' : 0 })
med_ds = pd.get_dummies(med_ds, columns=['region'], drop_first=True)


#checking for duplicates
#print("Number of duplicate records:", med_ds.duplicated().sum())
#print("Duplicate records:", med_ds.duplicated())

#med_ds = med_ds.drop_duplicates()
#print("New dataset after removing duplicates:", med_ds.shape)

med_ds.to_csv(r"C:\Users\tvhat\OneDrive\Desktop\VS CODE\Python\Medical Insurance\medical_insurance_cleaned.csv", index=False)


#understanding the shape of the dataset by visualizations
sns.histplot(med_ds['age'], kde=True, color='grey')
plot.title("Age Distribution")
plot.show()

sns.histplot(med_ds['bmi'], kde=True, color = 'skyblue')
plot.title("BMI Distribution")
plot.show()

sns.histplot(med_ds['charges'], kde=True, color="red")
plot.title("Medical Charges")
plot.show()

sns.boxplot(x='smoker', y='charges', data=med_ds)
plot.title("Charges based on Smoking Habits")
plot.show()

sns.boxplot(x='sex', y='charges', data=med_ds)
plot.title("Charges based on Gender")
plot.show


#Visualizing features present in the dataset
correlation = med_ds.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
plot.title("Feature Correlation Matrix")
plot.show()

#Visualizing how features are interacting with each other
sns.pairplot(med_ds, hue="smoker")
plot.suptitle("Pairplot: Interactions between features", y=1.02)
plot.show()

#Feature Engineering
#For BMI category

med_ds['bmi_category'] = pd.cut(med_ds['bmi'], bins = [0, 18.5, 25, 30, 100], 
                                labels= ['underweight', 'normal', 'overweight', 'obese'])

med_ds['age_category'] = pd.cut(med_ds['age'], bins=[17,30,50,65], labels=['Young', 'Middle-aged', 'Senior'])

med_ds['risk_fact'] = med_ds['smoker'] * med_ds['bmi']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
med_ds[['age','bmi']] = scaler.fit_transform(med_ds[['age', 'bmi']])

X = med_ds.drop(['charges'], axis=1)
y = med_ds['charges']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

med_ds.to_csv(r"C:\Users\tvhat\OneDrive\Desktop\VS CODE\Python\Medical Insurance\medical_insurance_cleaned.csv", index=False)

#Performing Kmeans clustering to make patient clusters
cluster_features  = ['age', 'bmi', 'smoker']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled_features = scaler.fit_transform(med_ds[cluster_features])

from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters = 3, random_state = 42)
clusters = kmeans.fit_predict(X_scaled_features)

med_ds['cluster'] = clusters

cluster_labels = {
    0: "High-Risk Smokers",
    1: "Average-Risk Non-Smokers",
    2: "Low-Risk Healthy Individuals"
}

#Map numeric cluster IDs to descriptive names
med_ds['cluster_label'] = med_ds['cluster'].map(cluster_labels)

#number of members in each cluster
print("\nCluster Counts:")
print(med_ds['cluster_label'].value_counts())

#Print cluster-wise summary statistics
print("\nCluster Summary by Label:")
summary_by_label = med_ds.groupby('cluster_label')[cluster_features + ['charges']].mean()
print(summary_by_label)

#PCA to reduce to 2 dimensions for plotting
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled_features)

#Plot the clusters using PCA
plot.figure(figsize=(10,6))
scatter = plot.scatter(
    x=pca_features[:,0],
    y=pca_features[:,1],
    c=clusters,
    cmap='viridis',
    alpha=0.6
)
plot.xlabel('PCA 1')
plot.ylabel('PCA 2')
plot.title("Patient KMeans Clusters (PCA View)")
plot.colorbar(label='Cluster')
plot.show()