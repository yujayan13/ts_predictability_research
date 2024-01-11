import netCDF4
import numpy
import xarray
import glob

#ensembles were downloaded in decades, this code combines these to have each ensemble be from 1950-2014
l=glob.glob(r"C:\ts_research\sst_ensembles\*.nc")
pre=[]
for f in l:
    p=f[:-17]
    if not(p in pre):
        pre.append(p)

for p in pre:
    ds = xarray.open_mfdataset(p+"*.nc",combine = 'nested', concat_dim="time")
    ds.to_netcdf(p+"_combined.nc")
