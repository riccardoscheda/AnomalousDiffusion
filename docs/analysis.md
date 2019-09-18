## Analysis.py

This library computes analysis on the images and on the fronts found with the library `fronts.py`. Mainly is needed to compute the area between the fronts of the cells and to compute the Mean Square Displacement (MSD) and Velocity Autocorrelation Function (VACF) to study the diffusion process of cell migration.

## `area(sx,dx)`

Computes the area of the polygon formed by the two borders of the cells

Parameters:
-----------------------------

-sx : pandas dataframe with the coordinates of the left border
-dx : pandas dataframe with the coordinates of the right border

Returns:

-the polygon formed by the two fronts (see the reference for the [Shapely Documentation](https://shapely.readthedocs.io/en/latest/manual.html#geometric-objects))

-the area between the two fronts (float)

#### Example

```
import cv2

import fronts as fr
import analysis as an


coin = cv2.imread("docs/images/histcoin.png")
coin = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

dfs , coordinates , binary = fr.fast_fronts(coin,size = 20,bands = False,length_struct=0,iterations = 1)

dx = dfs[0]
sx = dfs[1]

polygon, area = an.area(sx,dx)

```

## `error`

Computes the error between two arrays of areas in L^2


Parameters:
------------------------------------
area : array of floats
area_hand : array of floats

Returns an array with errors for each frame


## References
[1] [Shapely Documentation](https://shapely.readthedocs.io/en/latest/manual.html#geometric-objects)

[2]
