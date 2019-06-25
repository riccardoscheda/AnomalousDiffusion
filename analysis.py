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
    #read the file and put the coordinates into a Dataframe
    pol = pd.DataFrame(pd.read_csv(fname,delimiter =' '))
    #makes an object polygon in order to compute the area
    pol = np.array(pol)
    pol = Polygon(pol)
    #rotation of the polygon to have the right side, but not so important, is only for
    #visualization
    rotated = rotate(pol,180)
    reflected =  scale(rotated, xfact = -1)

    #returns the polygon and its area
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
    #makes an object polygon in order to compute the area
    pol0 = np.array(pol0)
    pol0 = Polygon(pol0)

    polsx = pd.DataFrame(pd.read_csv(path + "Sham_8-2-18_Field 5_1_sx.txt",sep ='\t'))
    polsx.columns = ["y","x"]
    poldx = pd.DataFrame(pd.read_csv(path + "Sham_8-2-18_Field 5_1_dx.txt",sep ='\t'))
    poldx.columns = ["y","x"]
    #makes an object polygon in order to compute the area
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
        #normalize the areas with respect to the area of the first frame
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
        #normalize the areas with respect to the area of the first frame
        areas_hand.append(pol2.area/pol1.area)
    #returns the two arrays with the normalized areas
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
    #computes the error in L^2 between the two areas
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

    Returns a pandas dataframe with the coordinates of the smoothed borders
    """
    #variables for the intervals of the grid
    min_x = 0
    max_x = l
    delta = l/float(N)

    #reads the file with the coordinates of the fronts
    grid = pd.DataFrame(pd.read_csv(file,delimiter = " "))
    grid.columns = [0,1]
    grid = grid.sort_values(by = 1)
    #makes a new aray where will be the smoothed fronts
    grid_smooth=np.empty((2,N))

    #makes the new grid taking the maximum value of the y coordinate in the interval
    for i in np.arange(N):
        #intervals
        bin_m = min_x +i*delta
        bin_M = min_x +(i+1)*delta
        #difference from sx front and dx front: in one case i take the max, in the
        #other i take the min
        if file.endswith("sx.txt"):
            grid_smooth[1,i]=min_x +(i+1/2.)*delta
            #takes the max value only if in the interval there are values
            if len(grid[(grid[1]>bin_m) & (grid[1]<bin_M)][0]) != 0:
                grid_smooth[0,i]=max(grid[(grid[1]>bin_m)&(grid[1]<bin_M)][0])
            else:
            #if there are not points in that interval, takes the latest value computed
                grid_smooth[0,i] = grid_smooth[0,i-1]
            if i == 0:
                grid_smooth[0,i] = np.mean(grid[0])
        else:
            grid_smooth[1,i]=min_x +(i+1/2.)*delta
            if len(grid[(grid[1]>bin_m)&(grid[1]<bin_M)][0]) != 0:
                grid_smooth[0,i]=min(grid[(grid[1]>bin_m)&(grid[1]<bin_M)][0])
            else:
                grid_smooth[0,i] = grid_smooth[0,i-1]
            if i == 0:
                grid_smooth[0,i] = np.mean(grid[0])

    df = pd.DataFrame()
    df[0] = pd.Series(grid_smooth[0])
    df[1] = pd.Series(grid_smooth[1])
    #return the dataframe with the smoothed grid
    return df

def velocity(df0, df1):
    """
    Computes the velocity of the fronts. Since the
    Delta t is defined by the single frame, the velocity of the front results to be
    only the difference between the final and the initial position between two frames

    --------------------------
    Parameters:
    --------------------------
    df0 : pandas dataframe which contains the x and y coordinates of the front of the initial frame
    df1 : pandas dataframe which contains the x and y coordinates of the front of the final frame

    ----------------------------
    Returns a dataframe with the velocity
    """
    velocity = df1[0] - df0[0]
    return velocity

def VACF(df0,df1):
    """
    Computes the Velocity Autocorrelation Function (VACF)
    which is the mean scalar product between two velocities

    --------------------------
    Parameters:
    --------------------------
    df0 : pandas dataframe which contains the x and y coordinates of the front of the initial frame
    df1 : pandas dataframe which contains the x and y coordinates of the front of the finale frame

    ----------------------------
    Returns a dataframe with the VACF
    """
    return sum(df0*df1)/len(df0)
