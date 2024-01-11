import glob
import xarray as xr
import numpy as np

from shapely.geometry import Point
import geopandas as gpd


#Load datasets
list_of_ts_ensembles = glob.glob(r"C:\ts_research\ts_ensembles\*.nc")
list_of_sst_ensembles = glob.glob(r"C:\ts_research\sst_ensembles\*.nc")

ts_ensembles_loaded = []
for i in list_of_ts_ensembles:
    ts_ensembles_loaded.append(xr.load_dataset(i))
    
sst_ensembles_loaded = []
for i in list_of_sst_ensembles:
    sst_ensembles_loaded.append(xr.load_dataset(i))
    

shapefile = gpd.read_file(r"C:\ts_research\cb_2022_us_nation_5m\cb_2022_us_nation_5m.shp")

#prepares the timeseries of the sst anomalies in the tropical pacific for october,november,december,january,feburary
def enso_anomaly_timeseries(ds_sst):
    #select nino 3.4 region
    region_TP = ds_sst.where(
        (ds_sst.lat <= 10) & (ds_sst.lat >= -10) & (ds_sst.lon >= 190) & (ds_sst.lon <= 260), drop=True
    )
    
    #calculate the average sst for the tropical pacific region for each month
    total_o = 0
    divider_o = 0
    total_n = 0
    divider_n = 0
    total_d = 0
    divider_d = 0
    total_j = 0
    divider_j = 0
    total_f = 0
    divider_f = 0
    
    for i in range(1950,2014):
        for t in region_TP.time.values:
            if str(t)[5:7]=="10" and str(t)[0:4]==str(i):
                total_o+=region_TP.sel(time = t)["SST"]
                divider_o+=1
            if str(t)[5:7]=="11" and str(t)[0:4]==str(i):
                total_n+=region_TP.sel(time = t)["SST"]
                divider_n+=1
            if str(t)[5:7]=="12" and str(t)[0:4]==str(i):
                total_d+=region_TP.sel(time = t)["SST"]
                divider_d+=1
            if str(t)[5:7]=="01" and str(t)[0:4]==str(i+1):
                total_j+=region_TP.sel(time = t)["SST"]
                divider_j+=1
            if str(t)[5:7]=="02" and str(t)[0:4]==str(i+1):
                total_f+=region_TP.sel(time = t)["SST"]
                divider_f+=1
    
    average_o = total_o/divider_o
    average_n = total_n/divider_n
    average_d = total_d/divider_d
    average_j = total_j/divider_j
    average_f = total_f/divider_f
    
    
    
    sst_anomalies = []
    #calculate anomalies using average
    for i in range(1950,2014):
        sst_o = 0
        sst_n = 0
        sst_d = 0
        sst_j = 0
        sst_f = 0
        for t in ds_sst.time.values:
            if str(t)[5:7]=="10" and str(t)[0:4]==str(i):
                sst_o+=region_TP.sel(time = t)["SST"]
            if str(t)[5:7]=="11" and str(t)[0:4]==str(i):
                sst_n+=region_TP.sel(time = t)["SST"]
            if str(t)[5:7]=="12" and str(t)[0:4]==str(i):
                sst_d+=region_TP.sel(time = t)["SST"]
            if str(t)[5:7]=="01" and str(t)[0:4]==str(i+1):
                sst_j+=region_TP.sel(time = t)["SST"]
            if str(t)[5:7]=="02" and str(t)[0:4]==str(i+1):
                sst_f+=region_TP.sel(time = t)["SST"]
        sst_anomalies.append(sst_o-average_o)
        sst_anomalies.append(sst_n-average_n)
        sst_anomalies.append(sst_d-average_d)
        sst_anomalies.append(sst_j-average_j)
        sst_anomalies.append(sst_f-average_f)
        
        
    return sst_anomalies

#selecting surface temperature within the US and for specific months
def US_ts(ds):
    us = ds.where(
        (ds.lat <= 50) & (ds.lat >= 24) & (ds.lon >= 235) & (ds.lon <= 295), drop=True
    )
    return us['TS']
def get_ts(ds_US,year,month):
    for i in ds_US.time.values:
        if str(i)[5:7]==month:
            if str(i)[0:4]==str(year+1):
                return(ds_US.sel(time = i))

#creating mask to select points within the US
ds_US_mask = US_ts(ts_ensembles_loaded[0]) #we only choose [0] because we want the coordinates only

lenx,leny = np.array(get_ts(ds_US_mask,1950,"03")).shape
lons,lats = np.meshgrid(ds_US_mask['lon'],ds_US_mask['lat'])


ds_mask = ts_ensembles_loaded[0].sel(time = ts_ensembles_loaded[0].time.values[0])["TS"]
ds_mask_us = ds_mask.where((ds_mask.lat <= 50) & (ds_mask.lat >= 24) & (ds_mask.lon >= 235) & (ds_mask.lon <= 295), drop = True)

#using the shapefile to select points within the US

for ind2,lat in enumerate(ds_mask_us.coords["lat"].values):
    for ind1,lon in enumerate(ds_mask_us.coords["lon"].values):
        lon = lon-360
        point = Point(lon,lat)
        is_within_shapefile = False
        for shape in shapefile.geometry:
            if point.within(shape):
                is_within_shapefile = True
                break
        if is_within_shapefile==False:
            ds_mask_us[ind2][ind1] = 0.0
        else:
            ds_mask_us[ind2][ind1] = 1.0


mask=(np.array(ds_mask_us)==0.0)
x,y = mask.shape
print(x)
print(y)



print("done")

list_us = []
list_of_months = ["03","04","05","06","07","08"]
for month in list_of_months:
    ml = []
    for i in ts_ensembles_loaded:
        ds_us = US_ts(i)      
        for j in range(1950,2014): 
            ml.append(np.ma.masked_array(get_ts(ds_us,j,month), mask = mask).flatten().compressed())
    list_us.append(ml)
    print("done")




#prepares the input data(tropical pacific region sst anomaly of several months) for each ensemble
list_sst = []
for i in sst_ensembles_loaded:
    list_sst.append(enso_anomaly_timeseries(i))

#22x57: size of tropical Pacific region for these ensembles, 5 for each month

sst_nino = np.array(list_sst).flatten().reshape(6400,5,22,57)

#save files for training
np.savez("training_march_ts.npz", enso = sst_nino, ts = list_us[0])
np.savez("training_april_ts.npz", enso = sst_nino, ts = list_us[1])
np.savez("training_may_ts.npz", enso = sst_nino, ts = list_us[2])
np.savez("training_june_ts.npz", enso = sst_nino, ts = list_us[3])
np.savez("training_july_ts.npz", enso = sst_nino, ts = list_us[4])
np.savez("training_august_ts.npz", enso = sst_nino, ts = list_us[5])

