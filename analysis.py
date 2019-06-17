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


def area_btw_fronts(df_sx,df_dx):
    """
    Computes the area between the two  different borders of the cells

    -----------------------------
    Parameters:
    -----------------------------

    df_sx : pandas Dataframe which contains x and y coordinates of the left side front
    df_dx : pandas Dataframe which contains x and y coordinates of the right side front
    """
    area_h = df_dx-df_sx
    #rimetto le x giuste in area_h
    area_h["x"]=df_dx["x"]
    col= area_h.columns
    x= list(area_h[col[0]])
    df= area_h[col[1:]]/2.
    area=(df.T).T.sum()
    return area

def error(front, df_sx, df_dx):
    """
    Computes the error between the areas (between the borders) with the fronts founded with
    pca and the fronts drawn by hand

    ------------------------------------
    Parameters:
    ------------------------------------
    front : the path of a txt file with the unique front
    df_sx : the path of a txt file the x and y coordinates of the left side front
    df_dx : the path of a txt file with the x and y coordinates of the right side front
    """
    pol2 = pd.DataFrame(pd.read_csv(df_sx,delimiter ='\t'))
    pol3 = pd.DataFrame(pd.read_csv(df_dx,delimiter ='\t'))

    pol2.columns = ["x","y"]
    pol3.columns = ["x","y"]
    pol2 = pol2.append(pol3)
    pol = np.array(pol3)
    pol = Polygon(pol)

    area2 = pol.area
    _ , area1 = area(front)

    return (area1 - area2)**2
