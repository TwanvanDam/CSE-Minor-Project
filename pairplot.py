#!/usr/bin/python3.12

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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

def ClusterAverage(df,save=True):
    df = pd.read_csv("./Results/merged_data_clean.csv")
    
    columns = df.columns
    df[columns[0]] = df[columns[0]].str.replace('Scan', '').str.split("_").str[0]

    # Group by the first column and calculate the average of other columns
    df_avg = df.groupby(columns[0]).mean()

    # Reset the index to make the first column a regular column
    df_avg = df_avg.reset_index()

    closest_rows = pd.DataFrame()
    for Scan, frame in df.groupby(columns[0]):
        for row in range(len(df_avg[columns[0]])):
            if df_avg.iloc[row,0] == Scan:
                argmin = cdist(df_avg.iloc[:, 1:],frame.iloc[:, 1:])[row,:].argmin()
                closest_rows = closest_rows._append(frame.iloc[argmin,:])

    # Reset the index of closest_rows
    closest_rows = closest_rows.reset_index(drop=True)
    if save:
        closest_rows.to_csv("./Results/ClusterAverage.csv", index=False)
    return closest_rows

def clean_all(filename, cleaning_dict,postfix):
        df = pd.read_csv(filename)
        initial_shape = df.shape
        
        # Delete VoxelCount Colum because bad material property
        df.drop("VoxelCount", axis=1, inplace=True)
        
        # Clean the data and save it to a new csv file
        df = Clean_data(df, cleaning_dict)
        save_name = "./Results/merged_data" + postfix + "cleaned.csv"
        df.to_csv(save_name, index=False)
        print(f"Cleaned data saved to: {save_name}")
        return df

def pairplot(df, postfix, Features_split = True ,Include_yield = True ,ClusterAvg= False):
    # Get the column names for easy access
    columns = df.columns

    # Calculate the cluster average and save it to a new csv file
    if ClusterAvg:
        df = ClusterAverage(df,save=False)
        filename_cluster = "./Results/ClusterAverage.csv"
        df.to_csv(filename_cluster, index=False)
        print("Cluster average saved to: ", filename_cluster)
        Features_split = False

    # Slice the dataframe to only include the wanted features
    if Include_yield:
        data = [1,2,3,4,-1] # -1 is the yield
    else:
        data = [1,2,3,4]

    # if Features_split is true, then the first column (Scan Name) is also included
    if Features_split:
        data = [0]+(data)
        df[columns[0]] = df[columns[0]].str.replace('Scan', '').str.split("_").str[0]
        wanted = [columns[i] for i in data]
    else:
        wanted = [columns[i] for i in data]

    sliced_df = df[wanted]

    # Plot the data
    if Features_split:
        g = sns.PairGrid(sliced_df,hue=columns[0])
        g.map_diag(sns.histplot,hue=None)
        g.map_lower(sns.scatterplot)
        g.map_upper(sns.kdeplot,hue=None)
        g.add_legend()
        plt.savefig("./Results/Pairplot_clusters" + postfix+".png")
        plt.show()
    elif ClusterAvg:
        sns.pairplot(sliced_df,diag_kind="hist")
        plt.savefig("./Results/Pairplot_clusters_avg" + postfix+ ".png")
        plt.show()
    else:
        sns.pairplot(sliced_df,diag_kind="kde")
        plt.savefig("./Results/Plots/Pairplots/Pairplot" + postfix+ ".png")
        plt.show()
    return

def individual_scatterplots(df,postfix,columns_to_plot):
    columns = df.columns
    if postfix == "_energy" or postfix == "_energyuncleaned":
        color = "r"
    plt.scatter(df[columns[columns_to_plot[0]]],df[columns[columns_to_plot[1]]],color=color)
    plt.xlabel(columns[columns_to_plot[0]])
    plt.ylabel(columns[columns_to_plot[1]])
    plt.title(columns[columns_to_plot[0]] + " vs " + columns[columns_to_plot[1]])
    plt.savefig("./Results/Plots/Scatterplots/Scatterplot" + postfix + columns[columns_to_plot[0]]+columns[columns_to_plot[1]] + ".png")
    plt.show()
    return

def individual_histogram(df,postfix,column_to_plot):
    columns = df.columns
    if postfix == "_energy" or postfix == "_energyuncleaned":
        color = "r"
    plt.hist(df[columns[column_to_plot]],color=color)
    plt.xlabel(columns[column_to_plot])
    plt.savefig("./Results/Plots/Histograms/Histogram" + postfix + columns[column_to_plot] + ".png")
    plt.show()
    return

if __name__ == "__main__":
    for filename in ["./Results/merged_data_energy.csv", "./Results/merged_data_offset.csv"]:
        if filename == "./Results/merged_data_offset.csv":
            postfix = "_offset"
        else:
            postfix = "_energy"
        
        cleaning_dict = {"SurfaceAreaDensity": 10, "EulerNumberDensity" : -1e-4,"MeanBreadthDensity" : -0.0015}
        if cleaning_dict == {}: # if no cleaning was done, then the name of the file is the same as the original
            postfix += "uncleaned"

        df = clean_all(filename,cleaning_dict , postfix)
        print(df.columns)
        #pairplot(df,postfix,Features_split=True,Include_yield=True,ClusterAvg=False)
        
        for i in range(1,5):
            individual_histogram(df,postfix,column_to_plot=i)
            individual_scatterplots(df,postfix,columns_to_plot=[i,-1])
        individual_histogram(df,postfix,column_to_plot=-1)