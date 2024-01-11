import glob
import xarray as xr

import numpy as np
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors

from cartopy import config
import cartopy.crs as ccrs

import matplotlib.ticker as mticker

from operator import add


import scipy.stats as measures
from sklearn import metrics

from shapely.geometry import Point
import geopandas as gpd


#load data
list_of_ts_ensembles = glob.glob(r"C:\ts_research\ts_ensembles\*.nc")
list_of_sst_ensembles = glob.glob(r"C:\ts_research\sst_ensembles\*.nc")
shapefile = gpd.read_file(r"C:\ts_research\cb_2022_us_nation_5m\cb_2022_us_nation_5m.shp")


ts_ensembles_loaded = []
for i in list_of_ts_ensembles:
    ts_ensembles_loaded.append(xr.load_dataset(i))
    
sst_ensembles_loaded = []
for i in list_of_sst_ensembles:
    sst_ensembles_loaded.append(xr.load_dataset(i))
    
 
def US_ts(ds):
    us = ds.where(
        (ds.lat <= 50) & (ds.lat >= 24) & (ds.lon >= 235) & (ds.lon <= 295), drop=True
    )
    return us['TS']
def get_ts(ds_west,year,month):
    for i in ds_west.time.values:
        if str(i)[5:7]==month:
            if str(i)[0:4]==str(year+1):
                return(ds_west.sel(time = i))
ds_US_mask = US_ts(ts_ensembles_loaded[0]) #we only choose [0] because we want the coordinates only

lenx,leny = np.array(get_ts(ds_US_mask,1950,"03")).shape
lons,lats = np.meshgrid(ds_US_mask['lon'],ds_US_mask['lat'])



ds_mask = ts_ensembles_loaded[0].sel(time = ts_ensembles_loaded[0].time.values[0])["TS"]
ds_ts_us = ds_mask.where((ds_mask.lat <= 50) & (ds_mask.lat >= 24) & (ds_mask.lon >= 235) & (ds_mask.lon <= 295), drop = True)


for ind2,lat in enumerate(ds_ts_us.coords["lat"].values):
    for ind1,lon in enumerate(ds_ts_us.coords["lon"].values):
        lon = lon-360
        point = Point(lon,lat)
        is_within_shapefile = False
        for shape in shapefile.geometry:
            if point.within(shape):
                is_within_shapefile = True
                break
        if is_within_shapefile==False:
            ds_ts_us[ind2][ind1] = 0.0
        else:
            ds_ts_us[ind2][ind1] = 1.0


mask=(np.array(ds_ts_us)==0.0)
x,y = mask.shape

months = ["march","april","may","june","july","august"]



res_m = {}
res_l = {}

for month in months:
    res_m[month] = np.load(fr"C:\ts_research\results_{month}_ts.npz")
    res_l[month] = np.load(fr"C:\ts_research\results_lin_{month}_ts.npz")

results_dict = {}



def transform(array):
    ds_sst = sst_ensembles_loaded[0].sel(time = sst_ensembles_loaded[0].time.values[0])["SST"]
    ds_sst_us = ds_sst.where((ds_sst.lat <= 50) & (ds_sst.lat >= 24) & (ds_sst.lon >= 235) & (ds_sst.lon <= 295), drop = True)
    ind = 0
    for i,j in enumerate(mask):
        for k,l in enumerate(j):
            if l==False:
                ds_sst_us[i][k]=array[ind]
                ind+=1
            else:
                ds_sst_us[i][k]=0
    return ds_sst_us



#functions to check for ENSO events
def check_nino(event):
    if np.mean(event)>1.5:
        return(True)
    else:
        return(False)

def check_nina(event):
    if np.mean(event)<-1.5:
        return(True)
    else:
        return(False)

for month in ["april","may"]:
    print(month)
    
    m = res_m[month]
    l = res_l[month]
    
    cnn_pred = m["pred"].reshape(691,960)
    lin_pred = l["pred"].reshape(691,960)
    true_ts = m["y_test"]
    
    x_cnn = m["x_test"][0]
    x_lin = l["x"][0]
    
    #creating composites of TS anomalies during ENSO events
    
    total_ml = [0]*691
    divider = 0
    ml_list = []
    for ind,event in enumerate(x_cnn):
        if check_nino(event):
            total_ml = list(map(add, total_ml, [i[ind] for i in cnn_pred]))
            divider+=1
            ml_list.append([i[ind] for i in cnn_pred])
    total_ml = [y/divider for y in total_ml]
    
    
    total_true = [0]*691
    divider = 0
    true_list = []
    for ind,event in enumerate(x_cnn):
        if check_nino(event):
            total_true = list(map(add, total_true, [i[ind] for i in true_ts]))
            divider+=1
            true_list.append([i[ind] for i in true_ts])
    total_true = [y/divider for y in total_true]
    
    
    total_lin = [0]*691
    divider = 0
    lin_list = []
    for ind,event in enumerate(x_lin):
        if check_nino(event):
            total_lin = list(map(add, total_lin, [i[ind] for i in lin_pred]))
            divider+=1
            lin_list.append([i[ind] for i in lin_pred])
    total_lin = [y/divider for y in total_lin]
    
    mean_cnn = list(np.mean(cnn_pred, axis = 1))
    mean_lin = list(np.mean(lin_pred, axis = 1))
    mean_true = list(np.mean(true_ts,axis=1))

    #anomalies
    am = np.subtract(total_ml,mean_cnn)
    at = np.subtract(total_true,mean_true)
    al = np.subtract(total_lin,mean_lin)
    
    #transform in order to graph
    transformed_true_cnn = transform(at)
    transformed_ml = transform(am)
    transformed_lin = transform(al)
    
    #calculate pattern correlation
    cnn_pattern_coeff = measures.pearsonr(am,at)[0]
    lin_pattern_coeff = measures.pearsonr(al,at)[0]
    print(cnn_pattern_coeff)
    print(lin_pattern_coeff)
    print((cnn_pattern_coeff-lin_pattern_coeff)/lin_pattern_coeff)

    #change 0 values to nan
    for i in range(len(transformed_ml)):
        for point in range(len(transformed_ml[i])):
            if transformed_ml[i][point]==0:
                transformed_ml[i][point] = np.nan 
    for i in range(len(transformed_true_cnn)):
        for point in range(len(transformed_true_cnn[i])):
            if transformed_true_cnn[i][point]==0:
                transformed_true_cnn[i][point] = np.nan 
    for i in range(len(transformed_lin)):
        for point in range(len(transformed_lin[i])):
            if transformed_lin[i][point]==0:
                transformed_lin[i][point] = np.nan 
    
    #creating figures
    colors = plt.cm.get_cmap('RdBu') 
    colors_reversed = colors.reversed() 
    
    ax = plt.axes(projection=ccrs.EckertIV(central_longitude=260))
    ax.set_extent([-124, -63, 25, 50], crs = ccrs.PlateCarree())
    
    plt.tight_layout()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='black', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.ypadding = 12
    gl.xpadding = 12
    gl.xlocator = mticker.FixedLocator([-120,-110,-100,-90,-80,-70])
    
    levels = np.linspace(-3, 3, 21)
    
    colorarray = ["midnightblue", "mediumblue", "blue", "royalblue", "steelblue", "deepskyblue", "aqua", "lightblue", "paleturquoise", "white", "white", "white", "lightpink", "pink", "salmon", "orange", "darkorange", "red", "firebrick","darkred", [0.184,0.035,0.035,1]]
    cmap = ListedColormap(colorarray)
    

    
    fig=plt.contourf(lons, lats, transformed_ml,
                    transform=ccrs.PlateCarree(),cmap=colors_reversed,vmin = -3, vmax = 3, levels=levels, extend = "both")
    
    font = {'fontname':'Helvetica','size':16}

    #plt.title("CNN Prediction Composite - La Nina", fontdict=font)
    if month == "april":
        letter = "(b)"
        Month = "April"
    elif month == "may":
        letter = "(e)"
        Month = "May"
    
    ax.coastlines()
    plt.title(f"{letter} CNN {Month} TS Prediction", fontdict = font)

    #plt.colorbar(fig,ticks=[-2.0,-1.8,-1.6,-1.4,-1.2,-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0]) 
    plt.tight_layout()
    plt.savefig(f'{Month}_TS_prediction_CNN_elnino.png',dpi = 300,bbox_inches='tight')

    plt.show()
    
    
    
    
    ax = plt.axes(projection=ccrs.EckertIV(central_longitude=260))
    ax.set_extent([-124, -63, 25, 50], crs = ccrs.PlateCarree())
    
    plt.tight_layout()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='black', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.ypadding = 12
    gl.xpadding = 12
    gl.xlocator = mticker.FixedLocator([-120,-110,-100,-90,-80,-70])
    
    
    levels = np.linspace(-3, 3, 21)
    
    #colorarray = ["midnightblue", "mediumblue", "blue", "royalblue", "steelblue", "deepskyblue", "aqua", "lightblue", "paleturquoise", "white", "white", "white", "lightpink", "pink", "salmon", "orange", "darkorange", "red", "firebrick","darkred", [0.184,0.035,0.035,1]]
    #cmap = ListedColormap(colorarray)
    
    fig=plt.contourf(lons, lats, transformed_true_cnn,
                  transform=ccrs.PlateCarree(),cmap=colors_reversed, vmin = -3, vmax = 3, levels=levels, extend = "both")
    
    font = {'fontname':'Helvetica','size':16}

    if month == "april":
        letter = "(a)"
        Month = "April"
    elif month == "may":
        letter = "(d)"
        Month = "May"
    
    ax.coastlines()
    plt.title(f"{letter} True {Month} TS", fontdict = font)
    #plt.colorbar(fig, ticks=[-2.0,-1.6,-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2,1.6,2.0], location = "bottom") #ticks=[-2.0,-1.8,-1.6,-1.4,-1.2,-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0]) 
    plt.tight_layout()
    plt.savefig(f'{Month}_TS_true_elnino.png',dpi = 300,bbox_inches='tight')


    plt.show()
    
    
    
    
    ax = plt.axes(projection=ccrs.EckertIV(central_longitude=260))
    ax.set_extent([-124, -63, 25, 50], crs = ccrs.PlateCarree())
    
    plt.tight_layout()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='black', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.ypadding = 12
    gl.xpadding = 12
    gl.xlocator = mticker.FixedLocator([-120,-110,-100,-90,-80,-70])
    
    levels = np.linspace(-3, 3, 21)
    
    #colorarray = ["midnightblue", "mediumblue", "blue", "royalblue", "steelblue", "deepskyblue", "aqua", "lightblue", "paleturquoise", "white", "white", "white", "lightpink", "pink", "salmon", "orange", "darkorange", "red", "firebrick","darkred", [0.184,0.035,0.035,1]]
    #cmap = ListedColormap(colorarray)
    
    fig=plt.contourf(lons, lats, transformed_lin,
                  transform=ccrs.PlateCarree(),cmap=colors_reversed, vmin = -3, vmax = 3, levels=levels, extend = "both")
    
    font = {'fontname':'Helvetica','size':16}
    if month == "april":
        letter = "(c)"
        Month = "April"
    elif month == "may":
        letter = "(f)"
        Month = "May"

    plt.title(f"{letter} MLR {Month} TS Prediction", fontdict = font)

    ax.coastlines()
    #plt.colorbar(fig, ticks=[-3.0,-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4,3.0], location = "bottom") 
    #[-2.5,-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0,2.5]
    #[-3.0,-2.7,-2.4,-2.1,-1.8,-1.5,-1.2, -0.9,-0.6, -0.3, 0, 0.3,0.6, 0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0]
    plt.tight_layout()
    plt.savefig(f'{Month}_TS_prediction_MLR_elnino.png',dpi = 300,bbox_inches='tight')

    plt.show()
    
    

    
