import pandas as pd
import pylab as plt
import numpy as np
df = pd.read_csv("/home/riccardo/Desktop/coordinates-11-02-19.txt",sep = " ")

#aggiungo la colonna booleana "valid" per determinare se il frame è valido o no, settando tutti i frame validi di default
df["valid"] = True
df

means = []
sqd = []
diff = pd.DataFrame()

#valuto ogni fronte con quello precedente, con due grandezze: la media delle differenze in x, e la distanza media di ogni punto
for i in range(2,100):
    j = 1
    #considero il primo fronte precedente che è valido
    while df.loc[(i-2*j)*100]["valid"] == False:
        j = j + 1
        
    #prendo le x e le y dei fronti
    x1 = df["x"][i*100:(i+1)*100].reset_index()
    x0 = df["x"][(i-2*j)*100:(i-1*j)*100].reset_index()  
    y1 = df["y"][i*100:(i+1)*100].reset_index()
    y0 = df["y"][(i-2*j)*100:(i-1*j)*100].reset_index()
    #calcolo le differenze nelle x e la distanza media
    diff =  x1 - x0  
    sd = np.sqrt((x1["x"]-x0["x"])**2 + (y1["y"]-y0["y"])**2)
    
    #le appendo in due liste
    sqd.append(sd.mean())
    means.append(diff["x"].mean())
    #se la differenza nele x è maggiore di un certo valore allora setto il frame corrente a "non valido"
    if abs(diff["x"].mean()) > 10:
        df["valid"][i*100:(i+1)*100] = False
        
        

for i in range(len(means)):
    if abs(means[i])>10:
        c = "r"
    else: 
        c = "b"
    plt.scatter(i,means[i], c=c)

plt.plot(means,c="b", label= "mean x difference")
plt.plot(sqd, label= "mean distance")
plt.legend()
plt.show()

plt.figure()
plt.xlim(0,1600)
plt.ylim(0,1200)

# plot dei frame anomali
side = 1
for i in range(25,29):
    if df.loc[side*i*100]["valid"] == True:
        c = "b"
    else:
        c = "r"
        
    plt.plot(df["x"][side*i*100:(side*i+1)*100],df["y"][side*i*100:(side*i+1)*100],c = c)
    
plt.show()