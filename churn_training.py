import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

Random_state = 42
dataset = pd.read_csv("customer_churn_dataset.csv")
dataset = dataset.drop("CustomerID", axis=1)
print(dataset.info())

class_balance = dataset["Churn"].value_counts(normalize=True) * 100
print(class_balance)
print(dataset.describe())
print(dataset.duplicated().sum())
dataset = dataset.dropna(subset=["Churn"])
print(dataset.isna().sum())

x = dataset.drop(columns="Churn")
y = dataset["Churn"]

numerical_col = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                 "Payment Delay", "Total Spend", "Last Interaction"]  # I won't scale age
# here I'll use Onehotenc for gender and the rest ordinal
ordinal_col = ["Subscription Type", "Contract Length"]
gender_col = ["Gender"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=Random_state, stratify=y)

plt.figure(figsize=(12, 12))
plt.title("Churn distribution (training set) ")
sns.countplot(x=y_train)
plt.xticks([0, 1], ["Stayed", "Churned"])
plt.xlabel("The classes")
plt.ylabel("Number of customers")
plt.show()

for col in numerical_col:
    print(f"Statistical Visualization of feature {col} in the training set")
    plt.figure(figsize=(8, 8))
    plt.title(f"Scatterplot for {col} to visualize its spread")
    sns.scatterplot(x=col, y=y_train, data=x_train)
    plt.xlabel(f"{col}")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title(f"Boxplot for {col} to carry out its statistical test")
    sns.boxplot(y=col, data=x)
    plt.xlabel(f"{col}")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title(f"Histogram for {col}")
    sns.histplot(
        data=x_train,
        x=col,
        kde=True,
    )
    plt.xlabel(f"{col}")
    plt.ylabel("Density")
    plt.show()


ordinal_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal encoder", OrdinalEncoder(
            categories=[
                ["Basic", "Standard", "Premium"],
                ["Monthly", "Quarterly", "Annual"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))]
)

gender_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

scaler_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("Std_scaler", StandardScaler())
    ]
)


preprocessor = ColumnTransformer(
    [
        ("ordinal", ordinal_pipe, ordinal_col),
        ("label", gender_pipe, gender_col),
        ("numerical", scaler_pipe, numerical_col)
    ]
)

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
