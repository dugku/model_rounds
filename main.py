import numpy as np
import pandas as pd
import pprint
from models import train_test, logistic, boosted_tree, results, bayes_model
import arviz as az
import matplotlib.pyplot as plt
import os

def main():
    df = read_file("rounds.csv")

    df_cl = process_predictor(df)
    df_dummies = process_others(df_cl)

    x, y = get_seperate_data(df_dummies)

    x_train, x_test, y_train, y_test = train_test(x, y)
    #thingy_dict = logistic(x_train, y_train, x_test, y_test)
    #accuracy, class_report, con_matrix, scores = boosted_tree(x_train, y_train, x_test, y_test)
    #pprint.pprint(thingy_dict)
    bayes_model(x, y)
    #results(accuracy, classReport=class_report, conMa=con_matrix, scores=scores)
    
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

    x = df.drop("round_end_reason", axis=1)
    y = df['round_end_reason']
    
    x_dummies = pd.get_dummies(x)
    

    processed_df = pd.concat([x_dummies, y], axis=1)
    return processed_df

def get_seperate_data(df):
    x = df.drop("round_end_reason", axis=1)
    y = df['round_end_reason']

    return x, y

def save_plots(trace, outdir="./plots"):
    os.makedirs(outdir, exist_ok=True)

    az.plot_trace(trace, var_names=["intercept", "beta"], compact=True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/trace.png", dpi=200, bbox_inches="tight")
    plt.close()

    az.plot_posterior(trace, var_names=["intercept", "beta"], point_estimate="mean", hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig(f"{outdir}/posterior.png", dpi=200, bbox_inches="tight")
    plt.close()

    az.plot_forest(trace, var_names=["beta"], combined=True, credible_interval=0.95)
    plt.tight_layout()
    plt.savefig(f"{outdir}/forest_beta.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Diagnostics—optional but useful
    az.plot_energy(trace); plt.savefig(f"{outdir}/energy.png", dpi=200, bbox_inches="tight"); plt.close()
    az.plot_rank(trace, var_names=["intercept", "beta"]); plt.savefig(f"{outdir}/rank.png", dpi=200, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    main()