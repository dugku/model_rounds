from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV,cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xb
from xgboost.callback import EarlyStopping
import pprint
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xb
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt


def train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    return x_train, x_test, y_train, y_test

def logistic(x_train, y_train, x_test, y_test):

    lr = LogisticRegression(
        solver="lbfgs",          # robust default for L2
        penalty="l2",
        C=1.0,
        max_iter=2000,
        random_state=42,
        class_weight="balanced"
    )

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", lr)
    ])

    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    cv_scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"  # area under PR curve (AP)
    }
    cv_res = cross_validate(
        pipe, x_train, y_train,
        scoring=cv_scoring,
        cv=cv,
        return_estimator=False
    )

    cv_summary = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v))
        } for k, v in cv_res.items() if k.startswith("test_")
    }

    pipe.fit(x_train, y_train)

    # Test predictions
    y_pred = pipe.predict(x_test)
    y_proba = pipe.predict_proba(x_test)[:, 1]  # positive-class probabilities


    roc = roc_auc_score(y_test, y_proba)

    # Robust test metrics
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_test, y_proba)
    }

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Coefficients (note: these are w.r.t. standardized features)
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])
    return {
        "model": pipe,
        "cv_summary": cv_summary,                 # means/stds for accuracy/roc_auc/pr_auc on train (CV)
        "test_metrics": test_metrics,             # accuracy, ROC-AUC, PR-AUC, log-loss, brier
        "classification_report": report,
        "confusion_matrix": cm,
        "test_pred_labels": y_pred,
        "test_pred_proba": y_proba,               # positive-class probs
        "intercept": intercept,
        "ROCscore" : roc    
    }

def boosted_tree(x_train, y_train, x_test, y_test):
    kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    spw = (neg / pos) if pos > 0 else 1.0

    param = {
        "n_estimators": 125    ,
        "objective": "binary:logistic",
        "max_depth": 6,
        "eval_metric": "logloss",
        "random_state": 42,
        "learning_rate": 0.10,
        "tree_method": "hist",
        "n_jobs": -1,
        "scale_pos_weight": spw,        
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    model = xb.XGBClassifier(**param)

    scores = cross_val_score(model, x_train, y_train, cv=kf, scoring="accuracy")
    
    model.fit(x_train, y_train)  
    
    y_pred = model.predict(x_test)

    y_scores = model.predict_proba(x_test)[:, 1]

    roc = roc_auc_score(y_test, y_scores)
    pr = average_precision_score(y_test, y_scores)
    print(f"XGB Test ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")


    class_report = classification_report(y_test, y_pred)

    con_matrix = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, class_report, con_matrix, scores

def bayes_model(x, y):
    """
    We want to model if the CT will win so, CTWin = 1 and TWin = 0.
    And somehow the CT and T win nis ~50 percent so we can model that as
    Bern
    """

    x = standardize_features(x)

    x = x.astype(np.float64)

    with pm.Model() as cs_round_model:
        intercept = pm.Normal('intercept', mu=0, sigma=1.5)

        beta = pm.Normal('beta', mu=0, sigma=0.5, shape=x.shape[1])

        log_odd = intercept + pm.math.dot(x, beta)

        p = pm.Deterministic('p', pm.math.sigmoid(log_odd))

        outcome = pm.Bernoulli("outcome", p=p, observed=y)

        trace = pm.sample(1000, tune=1000, chains=4, target_accept=0.95, return_inferencedata=True)

    return trace


def standardize_features(x):
    # Identify numeric columns (excluding already binary ones)
    numeric_cols = x.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numeric_cols if set(x[col].unique()).issubset({0, 1})]
    continuous_cols = [col for col in numeric_cols if col not in binary_cols]
    
    print(f"Continuous cols to standardize: {continuous_cols}")
    print(f"Binary cols to leave alone: {binary_cols}")
    
    if continuous_cols:
        scaler = StandardScaler()
        x_std = x.copy()
        x_std[continuous_cols] = scaler.fit_transform(x_std[continuous_cols])
        return x_std
    return x


def results(acc, classReport, conMa, scores):
    print(f"This is the accuracy of the model: {acc}\n")
    print(f"This is the classification report:\n {classReport}")
    print(f"Mean cross vidation scores: {scores.mean():.4f}")
    pprint.pprint(f"This is the confusion matrix: {conMa}")


