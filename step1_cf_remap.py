import multiprocessing
import os

import numpy as np
import pandas as pd
import xarray as xr
from cdo import *

cdo = Cdo()

### Extract the capacity factor data for the wind and solar resources in the ISOs
### !! no interpolation. just extract the points within each ISOs
global mdir
mdir = "/orcd/nese/mhowland/001/lyqiu/GODEEP/TGW/CF/"
ISOs = ["ERCOT", "ISONE", "CAISO"]
scenarios = ["rcp45hotter"]
startyear = {"historic": 2001, "future": 2020}
endyear = {"historic": 2020, "future": 2059}


def remapdata(odir, period, var, iy, maskfilen):
    input_fpath = f"{mdir}/{var}/{period}/{var}_gen_cf_{iy}.nc"
    print("Remapping input file:", input_fpath)
    ofile = f"{odir}/{var}_gen_cf_{iy}.nc"
    if not os.path.exists(ofile):
        data = cdo.ifthen(
            input=(
                f"{maskfilen} -remapnn,{maskfilen} "
                "-setgrid,/orcd/nese/mhowland/001/lyqiu/GODEEP/TGW_griddes "
                f"{input_fpath}"
            ),
            options="-b f32",
            returnXArray="capacity_factor",
        )
        if (
            "time" in data.dims
        ):  # if the time dimension is not named 'Time', rename it
            data = data.rename({"time": "Time"})
        time = pd.date_range(
            start=f"{iy}-01-01", end=f"{iy}-12-31 23:00", freq="h"
        )  # reformat the time dimension
        # remove feb 29
        time = time[~((time.month == 2) & (time.day == 29))]
        data["Time"] = time
        # The '365_day' calendar is used to handle datasets without leap years,
        # ensuring consistent time representation for climate data processing.
        encoding = {"Time": {"calendar": "365_day"}}
        data.to_netcdf(ofile, encoding=encoding)


if __name__ == "__main__":
    for ISO in ISOs:
        maskfilen = f"./ISOs/{ISO}_highres_remap.nc"
        print("Using mask file:", maskfilen)
        for var in ["wind", "solar"]:
            for period in scenarios:
                if period == "historic":
                    sy = startyear[period]
                    ey = endyear[period]
                else:
                    sy = startyear["future"]
                    ey = endyear["future"]
                odir = f"./data/{period}/{var}/{ISO}"
                print("Saving to output dir:", odir)
                if not os.path.exists(odir):
                    os.makedirs(odir)
                with multiprocessing.Pool(6) as pool:
                    for iy in range(sy, ey + 1):
                        results = {
                            iy: pool.apply_async(
                                remapdata,
                                args=(odir, period, var, iy, maskfilen),
                            )
                            for iy in range(sy, ey + 1)
                        }
