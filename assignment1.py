import numpy as np
import pandas as pd
from pathlib import Path
from pygam import LinearGAM, s, f

model = None
modelFit = None
pred = []

def load_csv(name, url):
    for p in (Path(name), Path.cwd() / name):
        if p.exists():
            return pd.read_csv(p)
    return pd.read_csv(url)

try:
    train = load_csv(
        "assignment_data_train.csv",
        "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
    )
    test = load_csv(
        "assignment_data_test.csv",
        "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
    )

    train["Timestamp"] = pd.to_datetime(train["Timestamp"])
    test["Timestamp"] = pd.to_datetime(test["Timestamp"])

    train["dow"] = train["Timestamp"].dt.dayofweek
    test["dow"] = test["Timestamp"].dt.dayofweek

    X_train = train[["hour", "dow", "month"]].values
    y_train = train["trips"].values
    X_test = test[["hour", "dow", "month"]].values

    model = LinearGAM(s(0, periodic=True) + f(1) + f(2))
    modelFit = model.fit(X_train, y_train)

    pred = modelFit.predict(X_test)
    pred = np.maximum(pred, 0).astype(float).tolist()
except Exception:
    pass
