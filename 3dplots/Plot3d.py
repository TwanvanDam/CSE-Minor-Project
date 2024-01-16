#!/usr/bin/python3.11

import os
import sys
import imageio.v2 as imageio
import matplotlib
matplotlib.use('TkAgg')  # Use a different backend, like TkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider
import numpy as np 
import plotly.graph_objects as go


def load_data(file_prefix,num_layers):
    #initialize array 
    bool_array = np.zeros((100,100,num_layers),dtype=bool)
    for i in range(0,num_layers):
        img = imageio.imread(f"{file_prefix}_{i:04d}.tif")
        bool_array[:,:,i] = img!=0 
    array_sum = np.sum(bool_array)
    return bool_array, array_sum

def clean_data(bool_array,downsample_size=1):
    bool_array = bool_array[::downsample_size,::downsample_size,::downsample_size]
    small_array = np.copy(bool_array)
    for z in range(1,bool_array.shape[2]-1): #z dimension
        for y in range(1,bool_array.shape[1]-1): #y dimension
            for x in range(1,bool_array.shape[0]-1): #x dimension
                if (bool_array[x,y,z])  and (bool_array[x-1,y,z]) and (bool_array[x+1,y,z]) and (bool_array[x,y+1,z]) and (bool_array[x,y-1,z]) and (bool_array[x,y,z-1]) and (bool_array[x,y,z+1]):
                    small_array[x,y,z] = False
    small_array_sum = np.sum(small_array)
    return small_array, small_array_sum

def plot_3d_volume(bool_array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    ax.voxels(filled=bool_array)
    plt.show()
    return
    
def plot_like_mri(vol):
    volume = vol.T
    print(volume.shape)
    r, c = volume[0].shape

    # Define frames
    import plotly.graph_objects as go
    nb_frames = 100

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(10 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[99 - k]),
        cmin=0, cmax=1
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=6.7 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[67]),
        colorscale='Gray',
        cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
        ))


    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             title='Slices in volumetric data',
             width=600,
             height=600,
             scene=dict(
                        zaxis=dict(range=[-0.1, 10], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    fig.show()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        file_prefix =  f"~/Downloads/20230206_Scan001_100px/Image Stacks/Scan{sys.argv[1]}/Scan{sys.argv[1]}"
    else:
        file_prefix = "~/Downloads/20230206_Scan001_100px/Image Stacks/Scan001_006/Scan001_006"

    if len(sys.argv) == 3:
        downsample_size = int(sys.argv[2])
    else:
        downsample_size = 5
    
    num_layers = 100
    data, array_sum = load_data(file_prefix,100)
    data_small, small_array_sum = clean_data(data,downsample_size)
    print(small_array_sum,array_sum)
    print(f"Decreased the datasize by {(1-small_array_sum/ array_sum)*100}%")
    #plot_like_mri(data)
    plot_3d_volume(data_small)
    
    
