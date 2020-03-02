#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:54:40 2019

@author: svitali
"""


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob as glob
import matplotlib.pylab as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
#get_ipython().magic(u'matplotlib inline')
import os
print(os.getcwd())
#os.chdir('/home/svitali/Documents/WORK/Fronts_presentation/')
#os.chdir('/home/riccardo/git/AnomalousDiffusion')
def necklace_points(points,N=1000,method='quadratic'):
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    # Interpolation for different methods:
    alpha = np.linspace(0, 1, N)
    interpolator =  interp1d(distance, points, kind=method, axis=0)
    return interpolator(alpha)


#%%
xcoords,ycoords={},{}
for side in ['dx','sx']:
    ############# corrisponde alla parte automatica di Riccardo
    #fnames= glob.glob('./fronti_manuali/Sham_8-2-18_Field 5_*_'+side+'.txt')
    #fnames=glob.glob('./fronti_manuali/EF 35V_21-2-18_Field 1_*_'+side+'.txt')
    fnames = "coordinates-20-11-18.txt"
    fronts={}
    df=pd.read_csv(fnames, sep=' ')
    #k=f.split('_')[-2]
    
    ###PARAMETRI DA MODIFICARE ######################
    points = 100
    field = 5
    ##########################################
    
    n_fronts = len(df)//points
    for i in range(50):
        df = df[df["side"]== side]
        df = df[df["field"]== field]
        fronts[i]=df[["x","y"]][i*100:(i+1)*100]
        #if fronts[i].T[1][0]>fronts[i].T[1][-1]:
         #   fronts[i]=fronts[i][::-1]

    
    curve={}
    method_name='quadratic'
    for k in fronts:
        # if fronts[k].T[1][0]>fronts[k].T[1][-1]:
        #     fronts[k]=fronts[k][::-1]
        curve[k]=necklace_points(fronts[k],method=method_name)

    xcoords[side] = pd.DataFrame.from_dict(dict((k, v.T[0]) for k, v in curve.items())).T.sort_index()
    ycoords[side] = pd.DataFrame.from_dict(dict((k, v.T[1]) for k, v in curve.items())).T.sort_index()
    ##################################fine
    #collasso le curve allo stesso punto di partenza
    xcoords[side]=xcoords[side]-xcoords[side][:1].values
    ycoords[side]=ycoords[side]-ycoords[side][:1].values
    #faccio riflessione per rendere sovrapponibili i fronti e semplificare analisi successive
    if side == 'dx':
        xcoords[side],ycoords[side]=-xcoords[side],-ycoords[side]


#%%
        
plt.plot(xcoords['dx'],'b:')
plt.plot(xcoords['sx'],'r--')
plt.xlabel('time')
plt.ylabel('displacement')
legend_elements = [Line2D([0], [0], color='b',linestyle=':', lw=1, label='right'),
                   Line2D([0], [0], linestyle='--', color='r',  lw=1, label='left')]
plt.legend(handles=legend_elements, loc='best')
#plt.savefig('sham_displacement.png')
#plt.savefig('EF_displacement.png')
plt.show()
#%%
# definisco un po' di funzioni utili e metodo di fit
#al metodo di fit sarebbe utile aggiungere negli output i parametri popt e salvarli entrambi
from scipy.optimize import curve_fit

def air_resistence_model(v,a,s):
    return a-v*s


def constant_acceleration_model(t,a,v=0,x=0):
    return x+v*t+a*t**2*.5

def elastic_membrane(t,w,p,a):
    return np.cos(w*t/10 + p)*a

def fit_my_model(y,func):
    #y=xcoords['dx'][0]
    x=np.arange(len(y))
    popt,pcov=curve_fit(func,x,y)
    ye=func(x,*popt)
    return ye
    #plt.plot(x,y,'bo')
    #plt.plot(x,ye,'k--')
    


#%% sottraggo alle traettorie il fit dell'andamento medio (drift)
confined={}

for side in ['dx','sx']:
    coords=xcoords[side]
    #fit_coords=coords.apply(lambda x: fit_my_model(x,constant_acceleration_model))
    fit_coords=fit_my_model(coords.T.mean().values,constant_acceleration_model)
    noise=(coords.T-fit_coords).T
    confined[side]=noise
    #plt.plot((noise**2).T.sum())
    if side=='dx':
        c='b:'
    else:
        c='r--'
    plt.plot(fit_coords,c)
    #plt.plot(noise,c)
plt.show() 

#plotto le traettorie senza il drift
plt.plot(confined['dx'],'b:',label='right')
plt.plot(confined['sx'],'r--',label='left')
plt.xlabel('time')
plt.ylabel('elastic component')
legend_elements = [Line2D([0], [0], color='b',linestyle=':', lw=1, label='right'),
                   Line2D([0], [0], linestyle='--', color='r',  lw=1, label='left')]
plt.legend(handles=legend_elements, loc='best')
#plt.savefig('sham_elastic_component.png')
#plt.savefig('EF_elastic_component.png')
plt.show()
  

#%% faccio il plot della media delle traettorie a cui ho sottratto il drift medio trovo una componente elastica
noise={}
for side in ['dx','sx']:
    coords=confined[side]
    #fit_coords=coords.apply(lambda x: fit_my_model(x,elastic_membrane))
    fit_coords=fit_my_model(coords.T.mean().values,elastic_membrane)
    noise[side]=(coords.T-fit_coords).T
    #plt.plot((noise**2).T.sum())
    if side=='dx':
        c='b:'
    else:
        c='r--'
    plt.plot(fit_coords,'k-')
    plt.plot(coords.T.mean(),c)
    
plt.show()
#faccio il plot delle traettorie a cui ho sottratto sia il drift che l'oscillazione media
#plt.plot(noise['dx'],'b:',label='right')
#plt.plot(noise['sx'],'r--',label='left')
plt.xlabel('time')
plt.ylabel('noise')
legend_elements = [Line2D([0], [0], color='b',linestyle=':', lw=1, label='right'),
                   Line2D([0], [0], linestyle='--', color='r',  lw=1, label='left')]
plt.legend(handles=legend_elements, loc='best')
#plt.savefig('sham_noise.png')
#plt.savefig('EF_noise.png')
plt.show()
#%%correlatione fluttuazione fronti dx e sx, vedo che le oscillazioni sono coerenti

plt.plot(confined['dx'].T.mean(),confined['sx'].T.mean(),'ko',markerfacecolor='w')
plt.xlabel('Right front elastic component')
plt.ylabel('Left front elastic component')
plt.show()
#%%#%% ensamble average traettorie
plt.plot(xcoords['dx'].T.mean(),'b:',label='right')
plt.plot(xcoords['sx'].T.mean(),'r--',label='left')
plt.legend()
plt.xlabel('time')
plt.ylabel('ensamble average displacement')
#plt.savefig('sham_average_displacement.png')
#plt.savefig('EF_average_displacement.png')
plt.show()
#%%ensamble average elastic component -> traettorie senza drift
plt.plot(confined['dx'].T.mean(),'b:',label='right')
plt.plot(confined['sx'].T.mean(),'r--',label='left')
plt.legend()
plt.xlabel('time')
plt.ylabel('ensamble average elastic component')
#plt.savefig('sham_average_fluctuation.png')
#plt.savefig('EF_average_fluctuation.png')
plt.show()
#%%#%%ensamble average noise -> traettorie senza drift e senza componente elastica
plt.plot(noise['dx'].T.mean(),'b:',label='right')
plt.plot(noise['sx'].T.mean(),'r--',label='left')
plt.legend()
plt.xlabel('time')
plt.ylabel('ensamble average noise')
#plt.savefig('sham_average_noise.png')
#plt.savefig('EF_average_noise.png')
plt.show()
#%%#%%ensamble average noise correlation left right 
# the correlation is lost after removing the average cosine fluctuation
plt.plot(noise['dx'].T.mean(),noise['sx'].T.mean(),'ko',markerfacecolor='w')
plt.xlabel('Right front noise')
plt.ylabel('Left front noise')
plt.show()
#%%#%%ensamble averaged MSD della componente stocastica -> no drift no componente elastica
#plt.plot((noise['dx'].T**2).mean(),'b:',label='right')
#plt.plot((noise['sx'].T**2).mean(),'r--',label='left')
import tidynamics as tidy
plt.plot(sum([tidy.msd(noise['dx'][c])for c in range(1000)])/1000,'b:',label='right')
plt.plot(sum([tidy.msd(noise['sx'][c])for c in range(1000)])/1000,'r--',label='left')
plt.legend()
plt.xlabel('time')
plt.ylabel('MSD')
plt.show()
#%%#%%ensamble averaged VACF
#plt.plot((noise['dx'].T*noise['dx'].T.values[0]).mean(),'b:',label='right')
#plt.plot((noise['sx'].T*noise['dx'].T.values[0]).mean(),'r--',label='left')
plt.plot(sum([tidy.acf(noise['dx'][c])for c in range(1000)])/1000,'b:',label='right')
plt.plot(sum([tidy.acf(noise['sx'][c])for c in range(1000)])/1000,'r--',label='left')
plt.legend()
plt.xlabel('time')
plt.ylabel('VACF')
