import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
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
print(dataset.isna().sum())

x = dataset.drop(columns="Churn")
y = dataset["Churn"]

numerical_col = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                 "Payment Delay", "Total Spend", "Last Interaction"]  # I won't scale age
# here I'll use labelenc for gender and the rest ordinal
ordinal_col = ["Subscription Type", "Contract Length"]
label_col = ["Gender"]

ordinal_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal encoder", OrdinalEncoder(
            categories=[
                ["Basic", "Standard", "Premium"],
                ["Monthly", "Quarterly", "Annual"]
            ]
        ))]
)

label_pipe = Pipeline(
    [
        ("label encoder", LabelEncoder())
    ]
)

scaler_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("Std_scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    [
        ("ordinal", ordinal_pipe, ordinal_col),
        ("label", label_pipe, label_col),
        ("numerical", scaler_pipe, numerical_col)
    ]
)

x = preprocessor.fit_transform(dataset)
print(x.head())

selection_scorings = [mutual_info_classif, chi2, f_classif]
mi_tech = {}
chi_tech = {}
f_tech = {}
try:
    for scoring in selection_scorings:
        pass
except Exception as e:
    print(e)
