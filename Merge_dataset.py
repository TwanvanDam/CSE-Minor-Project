#!/usr/bin/python3
import sys


for output_file in ["./Results/output_offset.txt", "./Results/output_energy.txt"]:
    output = open(output_file, 'r')
    dataset_name = output_file.split('/')[-1].split('_')[1].split('.')[0]
    merged = open("./Results/merged_data_" + dataset_name +".csv", 'w')

    output_lines = output.readlines()
    features = open("data/Microstructure Segmentation/All_extracted_features.csv", 'r')
    features_lines = features.readlines()

    merged.writelines(features_lines[0].replace("\n","")+", Yield Stress \n")

    for features_line in features_lines[1:]:
        for line in output_lines:
            line_clean = line.replace("\n", '').replace(" ", '').replace(".csv", "").split(',')
            if line_clean[0] == features_line.split(',')[0]:
                merged.writelines(features_line.replace("\n", "")+ "," +line_clean[1] + "\n")
    print("Merged data for " + dataset_name + " is ready")
    output.close()
    merged.close()
features.close()


