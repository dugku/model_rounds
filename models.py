from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xb

def train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    return x_train, x_test, y_train, y_test

def logistic(x_train, y_train, x_test, y_test):
    kf = KFold(n_splits=10, random_state=42, shuffle =True)

    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, x_train, y_train, cv=kf)
    
    model.fit(x_train, y_train)
    test_score = model.score(x_test, y_test)

    return  {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "test_score": test_score,
        "cv_scores": scores
    }


def boosted_tree(x_train, y_train, x_test, y_test):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    param = {
        "n_estimators": 250,
        "objective": "binary:logistic",
        "max_depth": 6,
        "eval_metric": "logloss",
        "random_state": 42,
        "learning_rate": 0.55,
        "tree_method": "hist"
    }

    model = xb.XGBClassifier(**param)

    scores = cross_val_score(model, x_train, y_train, cv=kf)

    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def bayes_model():
    """
    We want to model if the CT will win so, CTWin = 1 and TWin = 0.
    And somehow the CT and T win nis ~50 percent so we can model that as
    Bern
    """

    pass


def results():
    pass



