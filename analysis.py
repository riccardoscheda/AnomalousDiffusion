from shapely.affinity import rotate,scale
from shapely.geometry import Polygon
import os
import numpy as np
import matplotlib as plt
import cv2
import pandas as pd


def area(fname):
    """
    Computes the area of the polygon formed by the two borders of the cells

    -----------------------------
    Parameters:
    -----------------------------
    fname : the path of a txt file with the unique front

    -----------------------------
    References:
    -----------------------------
    [1] https://shapely.readthedocs.io/en/latest/manual.html#geometric-objects
    """

    pol = pd.DataFrame(pd.read_csv(fname,delimiter =' '))
    pol = np.array(pol)
    pol = Polygon(pol)
    rotated = rotate(pol,180)
    reflected =  scale(rotated, xfact = -1)

    return reflected,  reflected.area


def comparison():
    """
    Makes the comparison between the areas found by the fronts and the hand drawn fronts
    ------------------------------------------
    """
    path = "Data/data_fronts/"
    path1 = "Results/labelled_images1010/fronts/"

    #computes the areas for the first frame in order to normalize the other areas
    pol0 = pd.DataFrame(pd.read_csv(path1 + "fronts_labelled.m.0.png.txt",sep =' '))
    pol0 = np.array(pol0)
    pol0 = Polygon(pol0)

    polsx = pd.DataFrame(pd.read_csv(path + "Sham_8-2-18_Field 5_1_sx.txt",sep ='\t'))
    polsx.columns = ["y","x"]
    poldx = pd.DataFrame(pd.read_csv(path + "Sham_8-2-18_Field 5_1_dx.txt",sep ='\t'))
    poldx.columns = ["y","x"]

    polsx = polsx.append(poldx)
    polsx = np.array(polsx)
    pol1 = Polygon(polsx)


    areas = []
    areas_hand = []
    #computes the areas for all the frames
    for i in range(42):
        pol = pd.DataFrame(pd.read_csv(path1 + "fronts_labelled.m."+str(i)+".png.txt",sep =' '))
        pol = np.array(pol)
        pol = Polygon(pol)
        areas.append(pol.area/pol0.area)

        polsx = pd.DataFrame(pd.read_csv(path + "Sham_8-2-18_Field 5_"+str(i+1)+"_sx.txt",sep ='\t'))
        polsx.columns = ["y","x"]
        poldx = pd.DataFrame(pd.read_csv(path + "Sham_8-2-18_Field 5_"+str(i+1)+"_dx.txt",sep ='\t'))
        poldx.columns = ["y","x"]
        if poldx["x"][0]>100:
            poldx = poldx.reindex(index=poldx.index[::-1])
        if polsx["x"][0]<100:
            polsx = polsx.reindex(index=polsx.index[::-1])
        polsx = polsx.append(poldx)
        polsx = np.array(polsx)

        pol2 = Polygon(polsx)
        areas_hand.append(pol2.area/pol1.area)

    return np.array(areas) , np.array(areas_hand)

def error(area, area_hand):
    """
    Computes the error between the areas (between the borders) with the fronts founded with
    pca and the fronts drawn by hand

    ------------------------------------
    Parameters:
    ------------------------------------
    area : array with the areas found with the borders found with PCA
    area_hand : array with the areas found by the union of the hand drawn borders of the cells
    """
    error = np.sqrt((area - area_hand)**2)
    return np.array(error)


def grid(file, N = 100, l = 1200):
    """
    Makes an approximation of the fronts dividing the range in subintervals and for
    each subinterval takes the max value in that interval.

    --------------------------------------------
    Parameters:
    --------------------------------------------
    file: path of a txt file with x and y coordinates of the fronts
    N : number of subinterval to build the grid
    l : the length of the total interval
    """
    min_x = 0
    max_x = l
    delta = l/float(N)


    grid = pd.DataFrame(pd.read_csv(file,delimiter = " "))
    grid.columns = [0,1]
    grid = grid.sort_values(by = 1)
    grid_smooth=np.empty((2,N))

    for i in np.arange(N):
        bin_m = min_x +i*delta
        bin_M = min_x +(i+1)*delta
        if file.endswith("sx.txt"):
            grid_smooth[1,i]=min_x +(i+1/2.)*delta
            if len(grid[(grid[1]>bin_m) & (grid[1]<bin_M)][0]) != 0:
                grid_smooth[0,i]=max(grid[(grid[1]>bin_m)&(grid[1]<bin_M)][0])
            else:
                grid_smooth[0,i] = grid_smooth[0,i-1]
        else:
            grid_smooth[1,i]=min_x +(i+1/2.)*delta
            if len(grid[(grid[1]>bin_m)&(grid[1]<bin_M)][0]) != 0:
                grid_smooth[0,i]=min(grid[(grid[1]>bin_m)&(grid[1]<bin_M)][0])
            else:
                grid_smooth[0,i] = grid_smooth[0,i-1]
    df = pd.DataFrame()
    df[0] = pd.Series(grid_smooth[0])
    df[1] = pd.Series(grid_smooth[1])
    return df
