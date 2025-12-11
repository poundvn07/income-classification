import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess(df):
    df = df.drop(["fnlwgt", "education", "relationship"], axis=1)
    df["native.country"] = df["native.country"].apply(
        lambda x: "US" if x == "United-States" else "Non-US"
    )
    df.dropna(inplace=True)

    le = LabelEncoder()
    df["income"] = le.fit_transform(df["income"])

    df = pd.get_dummies(df, drop_first=True)

    num_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return df, X_train, X_test, y_train, y_test, scaler, num_cols
