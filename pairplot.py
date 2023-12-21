#!/usr/bin/python3.12

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def Clean_single_feature(df, column_name, value):
    if value < 0:
        df = df[df[column_name] >= value]
    else:
        df = df[df[column_name] <= value]
    return df

def Clean_data(df, cleaning_dict):
    for key in cleaning_dict.keys():
        df = Clean_single_feature(df, key, cleaning_dict[key])
    return df

df = pd.read_csv("./Results/merged_data.csv")
initial_shape = df.shape
columns = df.columns

Features_split = True
Include_yield = False

# set the lower bound for the features
cleaning_dict = {"SurfaceAreaDensity": 10, "EulerNumberDensity" : -1e-4,"MeanBreadthDensity" : -0.0015}

df = Clean_data(df, cleaning_dict)
new_shape = df.shape

print("Initial shape: ", initial_shape)
print("New shape: ", new_shape)
print("Removed: ", initial_shape[0] - new_shape[0], "rows")

df.to_csv("./Results/merged_data_clean.csv", index=False)

if Include_yield:
    data = [1,2,3,4,-1]
else:
    data = [1,2,3,4]

if Features_split:
    data = [0]+(data)
    df[columns[0]] = df[columns[0]].str.replace('Scan', '').str.split("_").str[0]
    wanted = [columns[i] for i in data]
else:
    wanted = [columns[i] for i in data]

sliced_df = df[wanted]
print(sliced_df.shape)

if Features_split:
    g = sns.PairGrid(sliced_df,hue=columns[0])
    g.map_diag(sns.histplot,hue=None)
    g.map_lower(sns.scatterplot)
    g.map_upper(sns.kdeplot,hue=None)
    g.add_legend()
    plt.savefig("./Results/Pairplot.png")
    plt.show()
else:
    sns.pairplot(sliced_df,diag_kind="kde")
    plt.savefig("./Results/Pairplot.png")
    plt.show()