#!/usr/bin/python3.12
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd

# Read data from file

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for file in ["./Results/merged_data.csv", "./Results/merged_data_linear.csv"]:
    #print column names
    data = pd.read_csv(file)
    columns = data.columns
    print(columns)
    Porosity = data["VolumeDensity"]

    second = "SurfaceAreaDensity"
    Surface = data[second]

    strength = data[" Yield Stress "]

    # Plot the data
    if file == "./Results/merged_data.csv":
        label = "Classic method"
    else:
        label = "Second derivative method"
    ax.scatter(Porosity[Surface<10], Surface[Surface<10], strength[Surface<10], marker='o',label=label)
plt.legend()
plt.title('Porosity vs Surface Area vs Yield Stress')
ax.set_xlabel('Porosity')
ax.set_ylabel(second)
ax.set_zlabel('Yield Stress')
plt.show()
