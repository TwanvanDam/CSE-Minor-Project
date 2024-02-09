'''
This file filters the data.txt file obtained from "GPs_Sklearn_our_dataset.py".
Input: data.txt
Output: cleaned data.txt (give it your own name)

Joey Cheung, Jan 2024
'''

with open(r'data.txt', 'r') as file:
    # Read each line from the file
    lines = file.readlines()

# Create a list to store filtered lines
filtered_lines = []

# Iterate through each line
for line in lines:
    if line[0].isdigit():
        # Split the line based on '|'
        values = line.split('|')
        
        # Extract the value between the 2nd and 3rd '|', convert it to float
        logp = float(values[2].strip())

        # Check if the value is below -700
        if logp > -380:
            # If it is, add the line to the filtered_lines list
            filtered_lines.append(line)

# Create a new file for writing the filtered lines
with open('new_test.txt', 'w') as output_file:
    # Write the filtered lines to the new file
    output_file.writelines(filtered_lines)