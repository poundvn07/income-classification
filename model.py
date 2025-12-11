import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv(
    'adult.csv',
    header=0,
    na_values='?',
    skipinitialspace=True
)

df = df.drop(['fnlwgt', 'education', 'relationship'], axis=1)
df['native.country'] = df['native.country'].apply(lambda x: 'US' if x == 'United-States' else 'Non-US')

print(df.head())
print(df.isna().sum())

num_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
cat_features = [col for col in df.columns if col not in num_features + ['income']]

df.dropna(inplace=True)

le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])

df = pd.get_dummies(df, drop_first=True)

X = df.drop(['income'], axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_proba = log_model.predict_proba(X_test)[:, 1]

Knn_model = KNeighborsClassifier(n_neighbors=15)
Knn_model.fit(X_train, y_train)
y_pred_knn = Knn_model.predict(X_test)
y_proba_knn = Knn_model.predict_proba(X_test)[:, 1]

def print_scores(model_name, y_test, y_pred, y_proba):
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print()

print_scores("LOGISTIC REGRESSION MODEL", y_test, y_pred, y_proba)
print_scores("K-NEAREST NEIGHBOURS", y_test, y_pred_knn, y_proba_knn)