import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import pprint
import numpy as np
from scipy.signal import find_peaks
import ipywidgets as wg
from IPython.display import display, HTML

CSS = """
.output {
    align-items: center;
}
"""

HTML('<style>{}</style>'.format(CSS))

def distance_cost_plot(distances):
    im = plt.imshow(distances, interpolation='nearest', cmap='Purples') 
    plt.gca().invert_yaxis()
    plt.xlabel("Reference")
    plt.ylabel("Target")
    plt.grid()
    plt.colorbar()
    
def euclidian_distances(time_seriesV, time_seriesH):
    distances = np.zeros([len(time_seriesV), len(time_seriesH)])
    for i in range(len(time_seriesV)):
        for j in range(len(time_seriesH)):
            distances[i,j] = (time_seriesH[j]-time_seriesV[i])**2
    return distances

def get_accumulated_cost(time_seriesV, time_seriesH, distances):
    accumulated_cost = np.zeros((len(time_seriesV), len(time_seriesH)))
    accumulated_cost[0,0] = distances[0,0]
    
    for i in range(1, len(time_seriesH)): 
        accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]
    for i in range(1, len(time_seriesV)):
        accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0]
    
    for i in range(1, len(time_seriesV)):
        for j in range(1, len(time_seriesH)):
            accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]
    
    path = [[len(time_seriesH)-1, len(time_seriesV)-1]]
    i = len(time_seriesV)-1 
    j = len(time_seriesH)-1
    
    while i > 0 or j > 0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append([j, i])
    path.append([0,0])
    
    path_x = [point[0] for point in path] # time_seriesH
    path_z = [point[1] for point in path] # time_seriesV
    paths = np.zeros([len(path_x), len(path_z)])
    
    return accumulated_cost, path_x, path_z, paths

def path_cost(time_seriesH, time_seriesV, accumulated_cost, distances):
    path = [[len(time_seriesH)-1, len(time_seriesV)-1]]
    cost = 0
    i = len(time_seriesV)-1
    j = len(time_seriesH)-1
    while i > 0 or j > 0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append([j, i])
    for [time_seriesV, time_seriesH] in path:
        cost = cost + distances[time_seriesH, time_seriesV]
    return path, cost

def plotDTWPaths(angles_vec_knee_ref, angles_vec_knee_target, time_row_cycle_ref, time_row_cycle_target, paths):
    plt.figure()
    for [map_row_cycle_ref, map_row_cycle_target] in paths:
        plt.plot([time_vec_cycle_target[map_row_cycle_target], time_vec_cycle_ref[map_row_cycle_ref]], [angles_vec_knee_target[map_row_cycle_target], angles_vec_knee_ref[map_row_cycle_ref]], 'r')
    plt.plot(time_vec_cycle_target, angles_vec_knee_target, color = 'b', label = 'Target', linewidth=5)
    plt.plot(time_vec_cycle_ref, angles_vec_knee_ref, color = 'g',  label = 'Reference', linewidth=5)
    plt.legend()
    plt.ylabel("Angle (degrees)")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.show()