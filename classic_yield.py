#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import os
import sys

offset = 0.0010

loading_vel = 1 #mm/min
sample_height = 1 #mm
sample_diameter = 1 #mm
# recording_interval = 2 #
strain_adjust = 0 #0.01647
area = np.pi*sample_diameter*sample_diameter/4

def get_yield(filename,offset=0.001,plot=True,save=False):
    data = np.genfromtxt(filename,delimiter=',',skip_header=1)

    strain = data[:,0]/sample_height
    stress = -data[:,1]*1000/area

    slope = (stress[10]-stress[0])/(strain[10]-strain[0])

    offset_stress = slope*(strain-offset)

    yield_index = np.argwhere(np.diff(np.sign(offset_stress-stress)))[0][0]

    yield_strain = strain[yield_index]
    yield_stress = stress[yield_index]
    if plot or save:
        plt.figure(figsize=(16,9),dpi=150)
        plt.plot(strain[:150],stress[:150],'.-',label="FEMdata")
        plt.plot(strain[:100],offset_stress[:100],label=f"{offset*100}% offset")
        plt.axvline(yield_strain,color="red",label="yield point")
        plt.title(filename.split("/")[-1].split(".csv")[0])
        plt.legend()
    if save:
        plt.savefig("./Results/Plots/Classic/" + filename.split("/")[-1].split(".csv")[0]+".png")
    if plot:
        plt.show()
    if not plot:
        plt.close()
    return yield_stress
    

cwd = os.getcwd()
data_dir = cwd + "/data/FEMResults/"
start = 0
end = len(os.listdir(data_dir))
i = 0

with open("output_classic.txt",'w') as output:
    with alive_bar(end-start) as bar:
        for csv_filename in os.listdir(data_dir):
            i += 1
            if (i > start) and (i <= end):
                if csv_filename != "Weird_Data":
                    yield_stress = get_yield(data_dir+csv_filename,plot=False,save=True)
                    output.writelines(f"{csv_filename.split("/")[-1]},{yield_stress}\n")
            bar()

if len(sys.argv) > 1:
    plot_file = sys.argv[1]

#get_yield(data_dir+"Scan"+plot_file+".csv",save=True)
