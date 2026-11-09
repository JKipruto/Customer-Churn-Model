import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif, chi2, f_classif
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("customer_churn_dataset.csv")
dataset.drop("CustomerID", axis=1)
print(dataset.info())

class_balance = dataset["Churn"].value_counts(normalize=True) * 100
print(class_balance)
print(dataset.describe())
print(dataset.duplicated().sum())

numerical_col = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                 "Payment Delay", "Total Spend", "Last Interaction"]
cat_col = ["Gender", "Subscription Type", "Contract Length"]
