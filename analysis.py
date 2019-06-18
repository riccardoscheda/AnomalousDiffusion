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


def error(area, area_hand):
    """
    Computes the error between the areas (between the borders) with the fronts founded with
    pca and the fronts drawn by hand

    ------------------------------------
    Parameters:
    ------------------------------------
    area : array with the areas found with the borders found with PCA
    area_hand : array dataframe with the areas found by the union of the hand drawn borders of the cells
    """
    error = np.sqrt((area - area_hand)**2)
    return error
