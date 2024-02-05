#!/usr/bin/python3
import sys
import pandas as pd

def open_file(output_file,features_file):
    #Read file containing the yield points
    output = open(output_file, 'r')
    
    #Assign the name of the dataset
    if "offset" in output_file:
        dataset_name = "offset"
    elif "energy" in output_file:
        dataset_name = "energy"
    else:
        dataset_name = "smth_went_wrong"
    if "final" in features_file:
        dataset_name = dataset_name + "_final"
    return output, dataset_name
    
def merge_datasets(output, dataset_name,features_file="../data/Microstructure Segmentation/All_extracted_features.csv"):
    #Merged will be the final file containing the features and the yield stress
    merged_name = "../Results/merged_data_" + dataset_name +".csv"
    merged = open(merged_name, 'w')
    
    #Read the yield points
    output_lines = output.readlines()
    
    #Read the features
    features = open(features_file, 'r')
    features_lines = features.readlines()
    merged.writelines(features_lines[0].replace("\n","")+",Yield Stress\n")

    #Merge the features and the yield stress
    for features_line in features_lines[1:]:
        for line in output_lines:
            line_clean = line.replace("\n", '').replace(" ", '').replace(".csv", "").split(',')
            if line_clean[0] == features_line.split(',')[0]:
                merged.writelines(features_line.replace("\n", "")+ "," +line_clean[1] + "\n")
    output.close()
    merged.close()
    features.close()
    return merged_name

def clean_dataset(filename, cleaning_dict):
    #read data to a dataframe
    df = pd.read_csv(filename)
    print(df.shape)
    #remove scans with a value of the feature higher than the threshold
    df.drop("VoxelCount", axis=1, inplace=True)

    for column_name, value in cleaning_dict.items():
        print(column_name, value)
        if value < 0:
            df = df[df[column_name] >= value]
                
        else:
            df = df[df[column_name] <= value]
    print(df.shape)
    return df

if __name__ == "__main__":
    cleaning_dict = {"SurfaceAreaDensity": 10}
    features_file = "../data/Microstructure Segmentation/All_extracted_features.csv"

    for output_file in ["../Results/output_offset.txt", "../Results/output_energy.txt"]:
        #merge the featues and the yield stress
        output, dataset_name = open_file(output_file,features_file)
        merged_name = merge_datasets(output, dataset_name,features_file)
        print("The dataset " + dataset_name + " has been merged and saved as " + merged_name)

        #clean the dataset
        df = clean_dataset(merged_name, cleaning_dict)
