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
    fname : the path of a txt file
    -----------------------------
    References:
    [1] https://shapely.readthedocs.io/en/latest/manual.html#geometric-objects
    """

    pol = pd.DataFrame(pd.read_csv(fname,delimiter =' '))
    pol = np.array(pol)
    pol = Polygon(pol)
    rotated = rotate(pol,180)
    reflected =  scale(rotated, xfact = -1)

    return reflected,  reflected.area
