#!/usr/bin/python3

features = open("data/Microstructure Segmentation/All_extracted_features.csv",'r')
output = open("Results/output.txt",'r')
merged = open("Results/merged_data.csv",'w')

output_lines = output.readlines()

features_lines = features.readlines()

merged.writelines(features_lines[0])

for features_line in features_lines[1:]:
    for line in output_lines:
        line_clean = line.replace("\n",'').replace(" ",'').replace(".csv","").split(',')
        if line_clean[0] == features_line.split(',')[0]:
            merged.writelines(features_line.replace("\n","")+","+line_clean[1]+"\n") 
    
