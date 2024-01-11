from __future__ import division

from scipy.stats import norm


import glob
import xarray as xr

import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cf

from shapely.geometry import Point
import geopandas as gpd
from matplotlib.colors import ListedColormap


plt.rc('font',family='arial')

#fisher z transformation and statistical significance test
def independent_corr(xy, ab, n, n2 = None, twotailed=True, conf_level=0.95, method='fisher'):
    xy_z = 0.5 * np.log((1 + xy)/(1 - xy))
    ab_z = 0.5 * np.log((1 + ab)/(1 - ab))
    if n2 is None:
        n2 = n

    se_diff_r = np.sqrt(1/(n - 3) + 1/(n2 - 3))
    diff = xy_z - ab_z
    z = abs(diff / se_diff_r)
    p = (1 - norm.cdf(z))
    if twotailed:
        p *= 2
    return z, p





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
    return us['PRECT']*86400000 #multiplied to change units from m/s to mm/day
def getmap(ds_west, year):
    for i in ds_west.time.values:
        if str(i)[5:7] =="03":
            if str(i)[0:4]==str(year+1):
                return(ds_west.sel(time = i))

#choose longitudes and latitudes
ds_lonlat = US_selection(ts_ensembles_loaded[0])
lons,lats = np.meshgrid(ds_lonlat['lon'],ds_lonlat['lat'])

#choose dataset for mask creation
ds_ts = ts_ensembles_loaded[0].sel(time = ts_ensembles_loaded[0].time.values[0])["TS"]
ds_ts_us = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)
ds_corr = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)
ds_linear = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)
ds_cnn = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)
#dataset for significant gridpoints
ds_sig = ds_ts.where((ds_ts.lat <= 50) & (ds_ts.lat >= 24) & (ds_ts.lon >= 235) & (ds_ts.lon <= 295), drop = True)

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




months = ["march","april","may", "june", "july","august"]



res_m = {}
res_l = {}

for month in months:
    #cnn results
    res_m[month] = np.load(fr"C:\ts_research\results_{month}_ts.npz")
    #mlr results
    res_l[month] = np.load(fr"C:\ts_research\results_lin_{month}_ts.npz")

results_dict = {}
cnn_dict = {}
lin_dict = {}
cnn_correlations = {}
lin_correlations = {}


for month in months:
    
    cnn_corr = res_m[month]["corr_coeffs"]
    lin_corr =res_l[month]["coeffs"]
    
    lin_dict[month] = np.mean(lin_corr)
    cnn_dict[month] = np.mean(cnn_corr)
    
    cnn_correlations[month] = (cnn_corr)
    lin_correlations[month] = (lin_corr)
    results_dict[month] = (cnn_corr-lin_corr)

ind_let = 0
letters = ["(a)", "(b)","(c)","(d)","(e)","(f)"]

for month in months:
    
    print(month)
    rmth = results_dict[month]
    
    #unflattening array of skill scores
    ind = 0
    for i,j in enumerate(mask):
        for k,l in enumerate(j):
            if l==False:
                ds_corr[i][k]=rmth[ind]
                ind+=1
            else:
                ds_corr[i][k]=0

    #making 0 points nan for computations    
    for i in range(len(ds_corr)):
        for point in range(len(ds_corr[i])):
            if ds_corr[i][point]==0:
                ds_corr[i][point] = np.nan

    
    cnn_corr = res_m[month]["corr_coeffs"]
    lin_corr =res_l[month]["coeffs"]
    
    ind = 0
    for i,j in enumerate(mask):
        for k,l in enumerate(j):
            if l==False:
                ds_linear[i][k]=lin_corr[ind]
                ind+=1
            else:
                ds_linear[i][k]=0
    
    ind = 0
    for i,j in enumerate(mask):
        for k,l in enumerate(j):
            if l==False:
                ds_cnn[i][k]=cnn_corr[ind]
                ind+=1
            else:
                ds_cnn[i][k]=0
    
    #significance test
    result = []
    for i in range(len(cnn_corr)):
        p = independent_corr(cnn_corr[i],lin_corr[i],960,960,twotailed = False, method = "fisher")[1]
        if p<=0.05 and cnn_corr[i]>lin_corr[i]:
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
    levels = np.linspace(-0.25, 0.25, 21)
    
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
    
    fig=plt.contourf(lons, lats, ds_corr,
                 transform=ccrs.PlateCarree(),cmap=cmap, vmin = -0.25, vmax = 0.25, levels=levels, extend = "both")
    
    longitudes = []
    latitudes = []
    for x in ds_sig:
        for y in x:
            if y==True:
                longitudes.append(y["lon"])
                latitudes.append(y["lat"])
        

    
    ax.coastlines()
    #plt.colorbar(fig, ticks=[-.25,-.2,-.15,-.10,-.05,0,.05,.10,.15,.20,.25],pad = 0.1,location = "bottom") 
    plt.tight_layout()
    
    #plotting dots for gridpoints with significant improvement
    fig = plt.scatter(longitudes, latitudes,
            color=[(0/255, 20/255, 0/255)],
            s=2,
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
    
    plt.savefig(f'{month}_correlation_colorbar.png',dpi = 600,bbox_inches='tight')
    plt.show()
    
    



linears = list(lin_dict.values())
cnns = list(cnn_dict.values())
num_pairs = len(cnns)

bar_width = 0.35
gap_width = 0.6
total_width = bar_width * 2 + gap_width + 0.1
x_values = np.arange(len(linears))*total_width

# Plotting the bar graph
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
plt.bar(x_values-0.05, cnns, width=bar_width, label='CNN', color = [0.03137254901960784, 0.3137254901960784, 0.5882352941176471])
plt.bar(x_values+bar_width+0.05, linears, width=bar_width, label='MLR', color = [0.4823529411764706, 0.7, 0.8686274509803922])

# Adding labels, title, and legend
plt.xlabel('Forecast Lead Time (months)',fontsize = 16, fontname='Arial')
plt.ylabel('Correlation Coefficient',fontsize = 16, fontname='Arial')
plt.legend(prop={'family': 'Arial', 'size': 16}, loc="upper center")




plt.xlim(-gap_width, max(x_values) + bar_width + gap_width)
plt.xticks(x_values + bar_width/2, np.arange(1, num_pairs + 1))
plt.yticks([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize = 12, fontname='Arial')
plt.tight_layout()
plt.savefig('correlation_bargraph_figure7.png',dpi = 600,bbox_inches='tight')
plt.show()
