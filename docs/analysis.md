## Analysis.py

This library computes analysis on the images and on the fronts found with the library `fronts.py`. Mainly is needed to compute the area between the fronts of the cells and to compute the Mean Square Displacement (MSD) and Velocity Autocorrelation Function (VACF) to study the diffusion process of cell migration.

## `area(sx,dx)`

Computes the area of the polygon formed by the two borders of the cells

#### Parameters:

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

## `error(area, area_hand)`

Computes the error between two arrays of areas in L^2


#### Parameters:

area : array of floats
area_hand : array of floats

Returns an array with errors for each frame


## `necklace_points(df,N=100,method='quadratic')`

Computes the interpolation based on the distance of x and y

#### Parameters:

-df : pandas dataframe which contains the x and y coordinates of a front

-N : number of intrpolation points

-method : the method for doing the interpolation

Returns:

-a pandas Dataframe with the interpolated coordinates (integers)

#### Example
```
import cv2

import fronts as fr
import analysis as an


coin = cv2.imread("docs/images/histcoin.png")
coin = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

dfs , coordinates , binary = fr.fast_fronts(coin,size = 20,bands = False,length_struct=0,iterations = 1)

dx = dfs[0]

interpolated_dx = an.necklace_points(dx, N = 1000)
```
we obtain a dataframe with the interpolated coordinates:
```
print(dx.tail())
print(interpolated_dx)
```
Old Dataframe:

 |x|y
---|---|---
558|...|...
559|...|...
560|444|838
561|439|838
562|438|839
563|436|839
564|436|840

New dataframe:

|x|y
---|---|---
993|...|...
994|...|...
995|439|837
996|438|838
997|437|839
998|436|838
999|436|840

## `VACF(df,conversion = "x")`
Computes the Velocity Autocorrelation Fuction (VACF)
which is the correlation  between the velocities of the fronts


#### Parameters:
-df : pandas dataframe with the coordinates

-conversion : string variable to convert pixels in micrometers, because the conversion is different for
the x and y axes

Returns a numpy array with the VACF

#### Example
```
import numpy as np
x = np.linspace(0,10,num=100)
df = pd.DataFrame()
df[0] = x
df[1] = x**2
df[2] = x**3
vacf = an.VACF(df)
plt.plot(vacf)
```

## `MSD(df, conversion = "x")`
Computes the Mean Square Displacement (MSD)
which is the mean squared difference between the x or y coordinates of the fronts


#### Parameters:

-df : pandas dataframe which contains x or y coordinates for different frames

-conversion : string variable to convert pixels in micrometers, because the conversion is different for
the x and y axes

Returns:

-a dataframe with the MSD

#### Example
```
import numpy as np
x = np.linspace(0,10,num=100)
df = pd.DataFrame()
df[0] = x
df[1] = x**2
df[2] = x**3
msd = an.MSD(df)
plt.plot(msd)
```

## `func(x,D,a)`

  The function for the fit of the Mean Squared Displacement (MSD)
  and the Valocity Autocorrelation Function (VACF)

## `fit(ydata)`
Returns the parameter D ( which is the diffusion coefficient) and a, which is the exponent which
tell us if the process is subdiffusive or superdiffusive
#### Parameters
  -ydata : array of floats

#### Example
```
import numpy as np
import pandas as pd

import analysis as an

x = np.linspace(0,10,num=100)
df = pd.DataFrame()
df[0] = x
df[1] = x**2
df[2] = x**3
vacf = an.VACF(df)

fit = an.fit(vacf.flatten())

fit
```
```
>>> array([6.72369147e-08, 6.26297029e+00])
```







## References
[1] [Shapely Documentation](https://shapely.readthedocs.io/en/latest/manual.html#geometric-objects)

[2] [tidynamics Documentation](http://lab.pdebuyl.be/tidynamics/)
