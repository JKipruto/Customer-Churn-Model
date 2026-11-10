import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix


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
