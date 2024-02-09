'''
This file makes a histogram that shows how many times each of the 4 feature ends up in the importance positions. 
This is done using the length scale hyperparameters.
Input: data.txt or the cleaned version as input (results of multiple GP models)
Output: histogram showing the importance positions of each feature

Joey Cheung, Jan 2024
'''

import numpy as np
import matplotlib.pyplot as plt

with open(r"newcleaned_manual_100_runs.txt", 'r') as file:
    # Read each line from the file
    lines = file.readlines()

feature1_list = []
feature2_list = []
feature3_list = []
feature4_list = []

for line in lines:
    if line[0].isdigit():
        # Split the second line based on the '|' character
        values_str = line.split('|')[1].strip()

        # Convert the space-separated values to a list of floats
        values_list = np.array([float(value) for value in values_str.split()])
        length_scales = values_list[1:-1]

        # Numpy array: the order of this array already implies the order of importance of the features
        # E.g. indices = [1, 0, 2, 3] means that variable 1 is the smallest, then variable 0, then variable 2, then variable 3
        # The index of our features starts at 1, so:
        # feature 2 is the smallest, then feature 1, then feature 3, then feature 4
        indices = np.argsort(length_scales)
        
        # Extract the importance position for feature 1. Same for features 2, 3 and 4
        feature1_list.append(np.where(indices == 0)[0][0])
        feature2_list.append(np.where(indices == 1)[0][0])
        feature3_list.append(np.where(indices == 2)[0][0])
        feature4_list.append(np.where(indices == 3)[0][0])
    else:
        continue

feature1_list = np.array(feature1_list)
feature2_list = np.array(feature2_list)
feature3_list = np.array(feature3_list)
feature4_list = np.array(feature4_list)

# Count how many times feature 1 ends up at importance position 1, 2, 3 and 4. Same for features 2, 3 and 4
feature1_pos1 = np.count_nonzero(feature1_list == 0); feature1_pos2 = np.count_nonzero(feature1_list == 1); feature1_pos3 = np.count_nonzero(feature1_list == 2); feature1_pos4 = np.count_nonzero(feature1_list == 3)
feature2_pos1 = np.count_nonzero(feature2_list == 0); feature2_pos2 = np.count_nonzero(feature2_list == 1); feature2_pos3 = np.count_nonzero(feature2_list == 2); feature2_pos4 = np.count_nonzero(feature2_list == 3)
feature3_pos1 = np.count_nonzero(feature3_list == 0); feature3_pos2 = np.count_nonzero(feature3_list == 1); feature3_pos3 = np.count_nonzero(feature3_list == 2); feature3_pos4 = np.count_nonzero(feature3_list == 3)
feature4_pos1 = np.count_nonzero(feature4_list == 0); feature4_pos2 = np.count_nonzero(feature4_list == 1); feature4_pos3 = np.count_nonzero(feature4_list == 2); feature4_pos4 = np.count_nonzero(feature4_list == 3)

# Make histogram
# Set width of bar 
barWidth = 0.15
fig = plt.subplots(figsize =(12, 8)) 
 
# Set height of bar 
feature1 = [feature1_pos1, feature1_pos2, feature1_pos3, feature1_pos4] 
feature2 = [feature2_pos1, feature2_pos2, feature2_pos3, feature2_pos4] 
feature3 = [feature3_pos1, feature3_pos2, feature3_pos3, feature3_pos4] 
feature4 = [feature4_pos1, feature4_pos2, feature4_pos3, feature4_pos4] 

# Set position of bar on X axis 
br1 = np.arange(len(feature1)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, feature1, color ='r', width = barWidth, 
        edgecolor ='grey', label ='feature1') 
plt.bar(br2, feature2, color ='g', width = barWidth, 
        edgecolor ='grey', label ='feature 2') 
plt.bar(br3, feature3, color ='b', width = barWidth, 
        edgecolor ='grey', label ='feature 3') 
plt.bar(br4, feature4, color ='y', width = barWidth, 
        edgecolor ='grey', label ='feature 4') 
 
# Adding Xticks 
plt.xlabel('Positions', fontsize = 15) 
plt.ylabel('Frequency', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(feature1))], 
        ['position 1', 'position 2', 'position 3', 'position 4'])
 
plt.legend()
plt.show() 