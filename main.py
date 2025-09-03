import numpy as np
import pandas as pd
import pprint
from models import train_test, logistic, boosted_tree, results


def main():
    df = read_file("rounds.csv")

    df_cl = process_predictor(df)
    df_dummies = process_others(df_cl)

    x, y = get_seperate_data(df_dummies)

    x_train, x_test, y_train, y_test = train_test(x, y)
    #thingy_dict = logistic(x_train, y_train, x_test, y_test)
    accuracy, class_report, con_matrix, scores = boosted_tree(x_train, y_train, x_test, y_test)
    #pprint.pprint(thingy_dict)
    results(accuracy, classReport=class_report, conMa=con_matrix, scores=scores)
    
def read_file(path):
    df = pd.read_csv(path)

    return df

def process_predictor(df):
    df["round_end_reason"] = df["round_end_reason"].map({
        "CTWin": 1, 
        "BombDefused" : 1,
        "TargetSaved": 1,
        "TWin": 0,
        "TargetBombed": 0
    })

    df_cl = df.dropna()

    return df_cl

def process_others(df):
    return pd.get_dummies(df)

def get_seperate_data(df):
    x = df.drop("round_end_reason", axis=1)
    y = df['round_end_reason']

    return x, y


if __name__ == "__main__":
    main()