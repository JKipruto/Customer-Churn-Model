import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
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
dataset["Churn"] = dataset["Churn"].astype(int)
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
    if col != "Support Calls":
        plt.figure(figsize=(6, 6))
        plt.title(f"Distribution of {col}")
        sns.kdeplot(
            data=x_train,
            x=col
        )
        plt.xlabel(col)
        plt.show()

    else:
        pass

    plt.figure(figsize=(8, 8))
    plt.title(f"Boxplot for {col} to carry out its statistical Analysis")
    sns.boxplot(y=col, data=x_train)
    plt.xlabel(col)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title(f"Histogram for {col}")
    sns.histplot(
        data=x_train,
        x=col,
        bins=10,
        hue=y_train,
        multiple="stack"
    )
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()
cat_col = {
    "Gender": ["Male", "Female"],
    "Subscription Type": ["Basic", "Standard", "Premium"],
    "Contract Length": ["Monthly", "Quarterly", "Annual"],
}

for col, order in cat_col.items():
    plt.figure(figsize=(6, 6))
    plt.title(f"Count for {col}")
    sns.countplot(x=col, data=x_train)
    plt.xlabel(col)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.title(f"Churn Rate by {col}")
    sns.barplot(
        x=col,
        y=y_train,
        data=x_train,
        order=order
    )
    plt.xlabel(col)
    plt.ylabel("Churn rate")
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


def model_pipe(model):  # I'll use this when training my models and performing CV
    return Pipeline([("Preprocessor", preprocessor), ("model", model)])


leaf_nodes = np.arange(10, 100, 5)
max_depth = np.arange(3, 30)
kernels = ["rbf", "linear", "sigmoid"]
gamma = ["auto", "scale"]
n_neighbors = np.arange(5, 10, 1)
weight = ["distance", "uniform"]
p = [1, 2]
voting_types = ["hard", "soft"]
n_estimators = np.arange(50, 100, 10)
learning_rate = np.logspace(-1, 1, 8)

voting_estimators = [("Destree", DecisionTreeClassifier(
    max_depth=100, max_leaf_nodes=10, random_state=Random_state)), ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", p=2))]

tuning_search = {
    "logreg": (LogisticRegression(random_state=Random_state), "grid", {"model__C": np.logspace(-2, 2, 10)}, None),
    "Destree": (DecisionTreeClassifier(random_state=Random_state), "grid", {"model__max_leaf_nodes": leaf_nodes, "model__max_depth": max_depth}, None),
    "svc": (SVC(random_state=Random_state), "grid", {"model__kernel": kernels, "model__gamma": gamma, "model__C": [0.1, 1, 10]}, None),
    "voting": (VotingClassifier(estimators=voting_estimators), "grid", {"model__voting": voting_types}, None),
    "knn": (KNeighborsClassifier(), "grid", {"model__n_neighbors": n_neighbors, "model__weights": weight, "model__p": p, }, None),
    "rf": (RandomForestClassifier(random_state=Random_state), "random", {"model__max_leaf_nodes": leaf_nodes, "model__max_depth": max_depth, "model__n_estimators": n_estimators}, 20),
    "bag": (BaggingClassifier(random_state=Random_state), "random", {"model__n_estimators": n_estimators}, 15),
    "adaboost": (AdaBoostClassifier(random_state=Random_state), "random", {"model__n_estimators": n_estimators, "model__learning_rate": learning_rate}, 15)
}

baseline_voting_estimators = [
    ("Destree", DecisionTreeClassifier(random_state=Random_state)),
    ("knn", KNeighborsClassifier()),
]


baseline_models = {
    "logreg": LogisticRegression(random_state=Random_state),
    "Destree": DecisionTreeClassifier(random_state=Random_state),
    "svc": SVC(random_state=Random_state),
    "knn": KNeighborsClassifier(),
    "voting": VotingClassifier(estimators=baseline_voting_estimators),
    "rf": RandomForestClassifier(random_state=Random_state),
    "bag": BaggingClassifier(random_state=Random_state),
    "adaboost": AdaBoostClassifier(random_state=Random_state),
}

x_tune, _, y_tune, _ = train_test_split(
    x_train, y_train, train_size=0.11, stratify=y_train, random_state=Random_state)

scorings = ["accuracy", "precision", "recall", "f1"]
values_list = []

for name, model in baseline_models.items():
    print(f"Running {name}...")
    cv_val = cross_validate(
        model_pipe(model),
        x_tune,
        y_tune,  # Switch to check if the model will perform better with larger number
        cv=3,
        scoring=scorings,
        n_jobs=-1
    )
    values_list.append(
        {
            "model": name,
            "accuracy": cv_val["test_accuracy"].mean(),
            "precision": cv_val["test_precision"].mean(),
            "recall": cv_val["test_recall"].mean(),
            "f1": cv_val["test_f1"].mean(),
            "f1_std": cv_val["test_f1"].std(),
            "fit_time_s": cv_val["fit_time"].mean()
        }
    )

baseline_df = pd.DataFrame(values_list).set_index(
    "model").sort_values("f1", ascending=False)
print(baseline_df.round(4))
models_tune = ["Destree", "rf", "voting", "bag"]  # ",svc", "adaboost", ]

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=Random_state
)

search_cv_est = {}
tuned_vals = {}
best_score = {}
for name, (model, kind, params, iter) in tuning_search.items():
    if name in models_tune:
        print(f"Running cross validation test for {name}")
        model = model_pipe(model)
        if kind == "grid":
            search_cv = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring="f1",
                cv=cv,
                n_jobs=2,
                verbose=2
            )

        else:
            search_cv = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=iter,
                scoring="accuracy",
                cv=cv,
                n_jobs=2,
                verbose=3,
                random_state=Random_state
            )
        search_cv.fit(x_tune, y_tune)
        search_cv_est[name] = search_cv.best_estimator_
        tuned_vals[name] = search_cv.best_params_
        best_score[name] = search_cv.best_score_

    else:
        pass

metrics = []
cm = {}

for name, model in search_cv_est.items():

    pred = model.predict(x_test)
    metrics.append(
        {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred)
        }
    )
    print(f"Classification report for {name}")
    print(classification_report(y_test, pred))
    cm[name] = confusion_matrix(y_test, pred)

for name, matrix in cm.items():

    plt.figure(figsize=(6, 6))
    plt.title(f"Confusion Matrix for {name}")
    sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
