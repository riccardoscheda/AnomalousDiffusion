import os
import numpy as np
import matplotlib as plt
import cv2
import pandas as pd
from hypothesis import given


import classification as cl
import fronts as fr
import analysis as an
filepath = "Data/"
test_image =  cv2.imread( filepath + "images/1.png")
im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

#######################################################
##TESTS FOR classification.py



def test_create_set():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir(filepath + "images/")) == 60

def test_create_modified_images():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir("Results/modified_images/")) == 59

def test_adaptive_contrast_enhancement():
    """
    Tests if the function returns a np.ndarray
    """
    assert isinstance(cl.adaptive_contrast_enhancement(im_gray,(10,10)), np.ndarray) == True

def test_LBP():
    """
    Tests:
    if the output histogram is normalized
    if the length of the histogram is 10
    if the output is a np.ndarray
    """
    assert isinstance(cl.LBP(im_gray), np.ndarray) == True
    assert sum(cl.LBP(im_gray)) <= 1.0
    assert sum(cl.LBP(im_gray)) >= 0.999
    assert len(cl.LBP(im_gray)) == 10

def test_Principal_components_analysis():
    """
    Tests:
    if the function returns a dataframe
    if does not return Nan values
    """
    assert isinstance(cl.Principal_components_analysis(im_gray), pd.DataFrame) == True
    assert any(np.where(np.isnan(cl.Principal_components_analysis(im_gray)))) == False

def test_classification():
    """
    Tests:
    if the output is numpy array
    if the output image is binary (there are only 0s and 1s)
    """
    binary = cl.classification(im_gray, cl.Principal_components_analysis(im_gray))
    assert isinstance(binary,np.ndarray) == True
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)

###############################################################################
#TESTS FOR fronts.py


def test_fronts():
    """
    Tests:
    if the output is a pandas DataFrame
    """
    fronts, _ = fr.fronts(filepath + "images/1.png")
    assert isinstance(fronts, pd.DataFrame) == True



def test_make_kernel():
    """
    Tests:
    if the output of make_kernel is a numpy matrix of 0s and 1s
    """
    struct = [0,0,0,1,0,0,0]
    binary = fr.make_kernel(struct, 1)
    assert isinstance(binary, np.matrix) == True
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)

def test_fast_fronts():
    """
    Tests :
    if the output is a list of two pandas dataframes
    if it saves two txt file for each input image
    """
    df, im, im2 = fr.fast_fronts("Results/modified_images/m_1.png", outdir = "")
    assert isinstance(df, list) == True
    assert len(df) == 2

def test_divide():
    """
    Tests:
    if return two pandas Dataframes
    if the two output dataframes are different
    """
    coord = pd.DataFrame(pd.read_csv("Results/labelled_images1216/fronts/fronts_labelled_m_1.png.txt" , delimiter=' '))
    coord.columns = ["x","y"]
    sx , dx = fr.divide(coord)
    assert len(sx) != 0
    assert len(dx) != 0
    assert isinstance(sx, pd.DataFrame) == True
    assert isinstance(dx, pd.DataFrame) == True
    assert all(sx["x"] == 0) == False
    assert all(dx["x"] == 0) == False
    assert sx.equals(dx) == False

##############################################################################
## TESTS FOR analysis.py


from shapely.affinity import rotate,scale
from shapely.geometry import Polygon


def test_area():
    """
    Tests:
    if returns a type Polygon and a number
    """
    path = "Results/labelled_images1216/fronts/fronts_labelled_m_0.png.txt"
    pol , area = an.area(path)
    assert isinstance(pol, Polygon) == True
    assert isinstance(area, float)


def test_comparison():
    """
    Tests:
    if returns two np.ndarrays
    if returns arrays with positive numbers
    """
    areas, areas_hand = an.comparison()
    assert all(areas >=0 ) == True
    assert all(areas_hand >=0 ) == True

import math

def test_error():
    """
    Tests:
    if returns an array of  positive numbers
    if it's commutative
    """


    #it shall be commutative ?
    a , b = an.comparison()
    error1 = an.error(a, b)
    error2 = an.error(b, a)
    assert all(error1 - error2 < 1e-5) == True
    assert isinstance(error1, np.ndarray) == True
    assert all(error1 > 0) == True

def test_grid():
    path = "Results/labelled_images1010/fronts/divided_fronts/fronts_labelled.m.2.png.txtsx.txt"
    N = 100
    l = 1200
    grid = an.grid(path, N , l)
    assert isinstance(grid, pd.DataFrame) == True
    assert len(grid) == N
def test_necklace_points():
    """
    Tests if:
    The output are two pandas Series
    the length of the two series are equal
    """
    path = "Results/modified_images/fronts/"
    fname = "m_33.png_sx.txt"
    dfx,dfy = an.necklace_points(path + fname, sep = " " )
    assert len(dfx) == len (dfy)
    assert isinstance(dfx, pd.Series) == True
    assert isinstance(dfy, pd.Series) == True


def test_velocity():
    """
    Tests:
    if the velocity is a DataFrame
    if the length of the velocity dataframe is equal to the length of the
    input DataFrames
    """
    path = "Results/labelled_images1010/fronts/divided_fronts/"
    fname0 = "fronts_labelled.m.10.png.txt"
    df0_sx = an.grid(path+fname0+"dx.txt")
    fname1 = "fronts_labelled.m.11.png.txt"
    df1_sx = an.grid(path+fname1+"dx.txt")
    vel = an.velocity(df0_sx,df1_sx)
    assert isinstance(vel, pd.Series) == True
    assert len(vel) == len(df0_sx)
    assert len(vel) == len(df1_sx)

def test_VACF():
    """
    Tests:
    if the output is a numpy array
    """
    path = "Data/data_fronts/"
    fname = "Sham_8-2-18_Field 5_"
    msd = an.VACF(path, fname, side = "_sx",delimiter = "\t")
    assert isinstance(msd, np.ndarray) == True

def test_MSD():
    """
    Tests:
    if the output is a numpy array
    if the output array elements are positive
    """
    path = "Results/modified_images/fronts/"
    pre = "m_"
    suf = ".png_sx.txt"
    msd, pos, y = an.MSD(path,nframes = 42,pre = pre, suf = suf,delimiter = " ")

    assert isinstance(msd, pd.DataFrame) == True
    assert all(msd.all() >= 0)

def test_MSD_Sham():
    """
    Tests:
    if the output is a numpy array
    if the output array elements are positive
    """
    path = "Data/data_fronts/"
    msd, pos2, y2 = an.MSD_Sham(path,side = "sx",delimiter = "\t")


    assert isinstance(msd, np.ndarray) == True
    assert np.isclose(msd[0], 0, rtol = 10e-5) == True
    assert all(msd > 0) == True
