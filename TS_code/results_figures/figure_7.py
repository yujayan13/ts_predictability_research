from __future__ import division

import glob
import xarray as xr

import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

from shapely.geometry import Point
import geopandas as gpd
from matplotlib.colors import ListedColormap
import cartopy.feature as cf

plt.rc('font',family='arial')


#load data
list_of_ts_ensembles = glob.glob(r"C:\ts_research\ts_ensembles\*.nc")
list_of_sst_ensembles = glob.glob(r"C:\ts_research\sst_ensembles\*.nc")

ts_ensembles_loaded = []
for i in list_of_ts_ensembles:
    ts_ensembles_loaded.append(xr.load_dataset(i))

sst_ensembles_loaded = []
for i in list_of_sst_ensembles:
    sst_ensembles_loaded.append(xr.load_dataset(i))
    
 
def US_selection(ds):
    us = ds.where(
        (ds.lat <= 50) & (ds.lat >= 24) & (ds.lon >= 235) & (ds.lon <= 295), drop=True
    )
    return us['TS']
def getmap(ds_west, year):
    for i in ds_west.time.values:
        if str(i)[5:7] =="03":
            if str(i)[0:4]==str(year+1):
                return(ds_west.sel(time = i))

#choose longitudes and latitudes
ds_lonlat = US_selection(ts_ensembles_loaded[0])
lons,lats = np.meshgrid(ds_lonlat['lon'],ds_lonlat['lat'])

#selecting US region for future use
ds_ts = ts_ensembles_loaded[0].sel(time = ts_ensembles_loaded[0].time.values[0])["TS"]
ds_ts_us = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)

ds_MSSS = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)
ds_l = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)

ds_sig = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)

#loading shapefile and building mask
shapefile = gpd.read_file(r"C:\ts_research\cb_2022_us_nation_5m\cb_2022_us_nation_5m.shp")
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

#definition of MSSS
def MSSS(predictions, observations):
    
    mean_squared_skill_score = []
    
    h = []
    observes = []
    for point in range(691):
        t = 0
        d = 0
        for ind,o in enumerate(observations[point]):
            t+=np.square((predictions[point][ind]-o))
            d+=1
        MSE_point = t/d
        
        total = 0
        divider = 0
        o_mean = np.mean(observations[point])
        for observe in observations[point]:
            total+=np.square((observe-o_mean))
            divider+=1
        MSE_O_point = total/divider
        msss_point = 1-(MSE_point/MSE_O_point)
        mean_squared_skill_score.append(msss_point)
        h.append(MSE_point)
        observes.append(MSE_O_point)
    
    overall = 1-(np.sum(h)/np.sum(observes))
    return mean_squared_skill_score, overall









months = ["march","april","may","june","july","august"]



res_m = {}
res_l = {}
bootstrap_res = {}

for month in months:
    #cnn results
    res_m[month] = np.load(fr"C:\ts_research\results_{month}_ts.npz")
    #mlr results
    res_l[month] = np.load(fr"C:\ts_research\results_lin_{month}_ts.npz")
    #results of bootstrapping
    bootstrap_res[month] = np.load(fr"C:\ts_research\bootstrap_{month}_ts.npz")
results_dict = {}
cnn_dict = {}
lin_dict = {}
entire_US_cnn = {}
entire_US_lin = {}
results_dict = {}
cnn_dict = {}
lin_dict = {}

p_val_dict = {}

for month in months:
    
    cnn_predictions = res_m[month]["pred"].reshape(691,960)
    lin_predictions =res_l[month]["pred"]
    cnn_observations = res_m[month]["y_test"]
    lin_observations = res_l[month]["y"]
    
    
    
    MSSS_cnn,overall_MSSS_cnn = MSSS(cnn_predictions,cnn_observations)
    MSSS_lin, overall_MSSS_lin = MSSS(lin_predictions,lin_observations)
    results_dict[month] = (np.array(MSSS_cnn)-np.array(MSSS_lin))
    cnn_dict[month] = MSSS_cnn
    lin_dict[month] = MSSS_lin
    entire_US_cnn[month] = overall_MSSS_cnn
    entire_US_lin[month] = overall_MSSS_lin
    
    p_val_dict[month] = np.array(bootstrap_res[month]["pval"])

ind_let = 0
letters = ["(a)", "(b)","(c)","(d)","(e)","(f)"]
for month in months:
    print(month)
    rmth = results_dict[month]
    
    ind = 0
    for i,j in enumerate(mask):
        for k,l in enumerate(j):
            if l==False:
                ds_MSSS[i][k]=rmth[ind]
                ind+=1
            else:
                ds_MSSS[i][k]=0


    
    cnn_MSSS = cnn_dict[month]
    lin_MSSS = lin_dict[month]
    
    
    result = []
    for i in range(len(cnn_MSSS)):
        p = p_val_dict[month][i]
        if p<=0.05 and cnn_MSSS[i]>lin_MSSS[i]:
            result.append(True)
        else:
            result.append(False)
    
    
    ind = 0
    for i,j in enumerate(mask):
        for k,l in enumerate(j):
            if l==False:
                ds_sig[i][k]=result[ind]
                ind+=1
            else:
                ds_sig[i][k]=0


    ax = plt.axes(projection=ccrs.EckertIV(central_longitude=255))
    ax.set_extent([-124, -70, 25, 50], crs = ccrs.PlateCarree())
    
    plt.tight_layout()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='black', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    
    gl.ypadding = 16
    gl.xpadding = 16
    
    
    levels = np.linspace(-0.15, 0.15,21)
   
    colorarray = [[0,10,50],
     [8,80,150],
     [8, 104, 172],
     [43, 140, 190],
     [78, 179, 211],
     [123, 204, 196],
     [168, 221, 181],
     [204, 235, 197],
     [230, 240, 240],
     [255, 255, 255],
     [255,255,255],
     [255, 255, 255], [255, 245, 204],[255,240,160], [255, 204, 51], [255, 153, 51], [255, 85, 0], [230, 40, 30], [200, 30, 20], [150,0,0],[100,0,0]]

    colorarray = [[a/255 for a in i] for i in colorarray]
    cmap = ListedColormap(colorarray)
    
    for i in range(len(ds_MSSS)):
        for j in range(len(ds_MSSS[i])):
            if ds_MSSS[i][j]==0:
                ds_MSSS[i][j]=np.nan
    for i in range(len(ds_MSSS)):
        for j in range(len(ds_MSSS[i])):
            if ds_MSSS[i][j]<-0.15:
                ds_MSSS[i][j]=0
    fig=plt.contourf(lons, lats, ds_MSSS,
                 transform=ccrs.PlateCarree(),cmap=cmap, vmin = -0.15, vmax = 0.15, levels=levels,extend = "both")
    
    
   
    longitudes = []
    latitudes = []
    for x in ds_sig:
        for y in x:
            if y==True:
                longitudes.append(y["lon"])
                latitudes.append(y["lat"])
        
    
    
    
    
    
    
    
    ax.coastlines()
    #plt.colorbar(fig, ticks=[-0.15,-0.12,-.09,-0.06,-0.03,0,0.03,.06,0.09,.12,0.15],pad = 0.1,location = "bottom") 
    plt.tight_layout()
    
    fig = plt.scatter(longitudes, latitudes,
            color="black",
            s=1.5,
            alpha=1,
            transform=ccrs.PlateCarree())
    
    
    font = {'size':24}
    font2 = {'size':10}

    ax.add_feature(cf.BORDERS)
    
    plt.gcf().set_size_inches(6.5, 3.7)
    
    title_month = month.capitalize()
    title_letter= letters[ind_let]
    ind_let+=1
    plt.title(f"{title_letter} {title_month}",fontdict=font)
    
    plt.savefig(f'{month}_MSSS.png',dpi = 600,bbox_inches='tight')

    plt.show()
