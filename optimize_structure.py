import gridpp
import numpy as np
import eccodes as ecc
import sys
import pyproj
import requests
import datetime
import argparse
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import fsspec
import os
import time
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import copy
import numpy.ma as ma
import warnings
import rioxarray
from flatten_json import flatten
from multiprocessing import Process, Queue
from fileutils import read_grib, write_grib
from obsutils import read_obs
from plot_output import plot
import netCDF4 as nc
from pyproj import Proj, transform
#from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import inspect

warnings.filterwarnings("ignore")

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--parameter_data", action="store", type=str, required=True)
    parser.add_argument("--dem_data", action="store", type=str, default="DEM_100m-Int16.tif")
    parser.add_argument("--dem_downs", action="store", type=str, default="elev_100m_1000m.nc")
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=True)

    args = parser.parse_args()

    allowed_params = ["temperature", "humidity", "windspeed", "gust"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)

    return args

def read_grid(args):
    """Top function to read "all" gridded data"""
    # Define the grib-file used as background/"parameter_data"
#    if args.parameter == "temperature":
#        parameter_data = args.t2_data
#    elif args.parameter == "windspeed":
#        parameter_data = args.ws_data
#    elif args.parameter == "gust":
#        parameter_data = args.wg_data
#    elif args.parameter == "humidity":
#        parameter_data = args.rh_data

    lons, lats, vals, analysistime, forecasttime = read_grib(args.parameter_data, True)

    _, _, topo, _, _ = read_grib(args.topography_data, False)
    _, _, lc, _, _ = read_grib(args.landseacover_data, False)

    # modify  geopotential to height and use just the first grib message, since the topo & lc fields are static
    topo = topo / 9.81
    topo = topo[0]
    lc = lc[0]

    if args.parameter == "temperature":
        vals = vals - 273.15
    elif args.parameter == "humidity":
        vals = vals * 100

    grid = gridpp.Grid(lats, lons, topo, lc)
    return grid, lons, lats, vals, analysistime, forecasttime, lc, topo

"""
def read_ml_grid(args):
    _, _, ws, _, _ = read_grib(args.ws_data, False)
    _, _, rh, _, _ = read_grib(args.rh_data, False)
    _, _, t2, _, _ = read_grib(args.t2_data, False)
    _, _, wg, _, _ = read_grib(args.wg_data, False)

    missing_data = 9999
    # check if any input grib_files contain missing data. If missing data when exit program
    all_input = {'ws': ws, 'rh': rh, 't2': t2, 'ws': ws, 'rh': rh, 't2': t2}
    for name, arr in all_input.items():
        if missing_data in arr:
            print(f"Missing data found in {name}")
            exit("Aborting program due to missing data.")

    # change parameter units:
    rh = rh * 100
    t2 = t2 - 273.15

    return ws, rh, t2, wg
"""

def interpolate(grid, points, background, obs, args, lc, obso, lon, lat):
    #griddem, points, background0[0], diff_point, args, lcdem
    """Perform optimal interpolation"""

    output = []

    # create a mask to restrict the modifications only to land area (where lc = 1)
    lc0 = np.logical_not(lc).astype(int)

    # Interpolate background data to observation points
    # When bias is gridded then background is zero so pobs is just array of zeros
    pobs = gridpp.nearest(grid, points, background)
    #print(pobs)

    # Barnes structure function with horizontal decorrelation length 30km,
    # vertical decorrelation length 200m

    

    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 5

    # error variance ratio between observations and background
    # smaller values -> more trust to observations
    obs_to_background_variance_ratio = np.full(points.size(), 0.1)
    # perform optimal interpolation
    
    # cross-validation
    #structure = gridpp.Barnes
    #structure_cv = gridpp.CrossValidation(structure, 2000)
    #gridpp.CrossValidation
    hh =  [30000, 30000, 30000]
    vv = [500, 400, 300] 
    kk = [0.5,0.5,0.5]
    for i in range(0, len(hh)):
        structure = gridpp.BarnesStructure(hh[i], vv[i], kk[i])
        #structure_cv = gridpp.CrossValidation(structure, 200)
        
        tmp_output = gridpp.optimal_interpolation(
        grid, # grid location class
        background, # grid values
        points, # obs point location class
        obs, # obs values
        obs_to_background_variance_ratio,
        pobs, # background values at obs points
        structure,
        max_points,
        )
        #print(tmp_output)
        obs_diff = gridpp.nearest(grid, points, tmp_output)
        plt.figure(figsize=(17, 6), dpi=80)
        plt.subplot(1, 3, 1)
        plt.scatter(
            obso["longitude"],
            obso["latitude"],
            s=10,
            c=obs,
            cmap="RdBu_r",
            vmin=(-5),
            vmax=5,
        )
        #"""
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="mnwc - obs " + "h " + args.parameter, orientation="horizontal"
        )
   
        plt.subplot(1, 3, 2)
        plt.scatter(
            obso["longitude"],
            obso["latitude"],
            s=10,
            #c=obs_diff["WS1bias"], 
            c=obs_diff,
            #c=ml_fcst[k]["biasc"],
            cmap="RdBu_r",
            vmin=(-5),
            vmax=5,
        )
        #"""
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="Diff to obs points " + "h " + args.parameter, orientation="horizontal"
        )

        plt.subplot(1, 3, 3)
        plt.pcolormesh(
            np.asarray(lon),
            np.asarray(lat),
            tmp_output,
            cmap="RdBu_r",
            vmin=-5,
            vmax=5,
        )
        #"""
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="Diff to obs points " + "h " + args.parameter, orientation="horizontal"
        )    
        # plt.show()
        plt.savefig("vv2_" + str(i) + args.parameter + ".png")
        #print("cv", cv_output.mean())

    return tmp_output

def main():
    args = parse_command_line()
    # print("Reading NWP data for", args.parameter )
    st = time.time()
    # read in the parameter which is forecasted
    # background contains mnwc values for different leadtimes
    grid, lons, lats, background, analysistime, forecasttime, lc, topo = read_grid(args)
    # create "zero" background for interpolating the bias
    background0 = copy.copy(background)
    background_orig = copy.copy(background)
    background0[background0 != 0] = 0

    et = time.time()
    timedif = et - st
    print(
        "Reading NWP data for", args.parameter, "takes:", round(timedif, 1), "seconds"
    )
    # Read observations from smartmet server
    # Use correct time! == latest obs hour ==  forecasttime[1]
    points, obs = read_obs(args, forecasttime, grid, lc, background, analysistime)

    ot = time.time()
    timedif = ot - et
    print("Reading OBS data takes:", round(timedif, 1), "seconds")

    # use 1km dem in gridding!
    #####
    nc_file = nc.Dataset(args.dem_downs, 'r')
    # Access latitude, longitude, and elevation data
    lat = nc_file.variables['y'][:]
    lon = nc_file.variables['x'][:]
    elev = nc_file.variables['Band1'][:]
      
    nc_file.close()
    lat = np.array(lat)
    lon = np.array(lon)
    elev = np.array(elev)

    lcdem = np.zeros_like(elev)
    # lat & lon are 1-d when elev is 2d, modify to 2d
    lon, lat = np.meshgrid(lon, lat)
    # convert Lambert Conformal Conic projection
    lon_0 = 15  # Central meridian
    lat_0 = 63.3  # Reference latitude
    lat_1 = 63.3  # First standard parallel
    lat_2 = 63.3  # Second standard parallel
    # Create a projection object for the Lambert Conformal Conic projection
    lcc_proj = Proj(proj='lcc', lon_0=lon_0, lat_0=lat_0, lat_1=lat_1, lat_2=lat_2)
    # Define the target coordinate system (decimal degrees)
    target_proj = Proj(proj='latlong', datum='WGS84')
    # Transform Lambert Conformal Conic coordinates to decimal degrees
    lon, lat = transform(lcc_proj, target_proj, lon, lat)
    
    """
    plt.figure(1)
    plt.contourf(lon, lat, elev, cmap='viridis')
    plt.colorbar(label='Elevation (m)')
    plt.title('Digital Elevation Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig("dem.png")
    """

    # create a grid class needed in gridpp
    griddem = gridpp.Grid(lat, lon , elev , lcdem)
    
    # do the downscaling to 1km
    if args.parameter == "temperature":
        # choose a constant gradient 
        gradient = -0.0065 # C/m
        #gradient = -0.65 # for grazy testing
        grid1km = gridpp.simple_gradient(grid, griddem, background, gradient)
    else:
        grid1km = gridpp.bilinear(grid, griddem, background)
    
    grid1km0 = copy.copy(grid1km)
    grid1km0[grid1km0 != 0] = 0
    background = grid1km
    background0 = grid1km0

    # Interpolate ML point forecasts for bias correction + 0h analysis time
    #print(len(ml_fcst))
    #print(ml_fcst[0].shape) 
    #print(ml_fcst[0].head(5))
    # exit()
    # origninal bg values to points
    fc_point = gridpp.nearest(grid, points, background_orig[1])
    # difference of obs minus bg to get the difference
    diff_point = fc_point - obs.iloc[:, 5]
    #print(diff_point)
    diff = interpolate(griddem, points, background0[0], diff_point, args, lcdem, obs, lon, lat)
    # Plot
    vmin = (-5)
    vmax =  5


    
    exit()
    # Assuming you have the true values in 'y_true' and the predicted values in 'y_pred'
    #r2 = r2_score(y_true, y_pred)
    #rmse = mean_squared_error(y_true, y_pred, squared=False)

    ##print("R2:", r2)
    #print("RMSE:", rmse)
    
    #oit = time.time()
    #timedif = oit - mlt
    #print("Interpolating forecasts takes:", round(timedif, 1), "seconds")
    # calculate the final bias corrected forecast fields: MNWC - bias_correction
    # and convert parameter to T-K or RH-0TO1
    print(obs.columns)
    output = []
    tmp_output = background[1] - diff
    # Implement simple QC thresholds
    if args.parameter == "humidity":
        tmp_output = np.clip(tmp_output, 5, 100)  # min RH 5% !
        tmp_output = tmp_output / 100
    elif args.parameter == "windspeed":
        tmp_output = np.clip(tmp_output, 0, 38)  # max ws same as in oper qc: 38m/s
    elif args.parameter == "gust":
        tmp_output = np.clip(tmp_output, 0, 50)
    elif args.parameter == "temperature":
        tmp_output = tmp_output + 273.15
    output = tmp_output

    # Remove analysistime (leadtime=0), because correction is not made for that time
    forecasttime.pop(0)
    #assert len(forecasttime) == len(output)
    # check for missing data in output
    if np.isnan(output).any() or np.any(output == None):
        print("Bias correction output contains NaN/None values")
        # replace nan/None with missing data in grib 9999
        output = np.where(np.isnan(output) | (output == None), 9999, output)
        # exit()
    #write_grib(args, analysistime, forecasttime, output)

    # plot the results
    vmin = np.amin(background_orig)
    vmax = np.amax(background_orig)
    print("vmin:", vmin)
    print("vmax:", vmax)

    vmin1 =  np.amin(output)
    vmax1 =  np.amax(output)
    print("vmin:", vmin1)
    print("vmax:", vmax1)

    obs_diff = gridpp.nearest(griddem, points, diff)
    plt.figure(figsize=(17, 6), dpi=80)
    plt.subplot(1, 4, 1)
    plt.pcolormesh(
        np.asarray(lons),
        np.asarray(lats),
        background_orig[1],
        cmap="Spectral_r",  # "RdBu_r",
        vmin=vmin,
        vmax=vmax
    )

    

    plt.xlim(0, 35)
    plt.ylim(55, 75)
    cbar = plt.colorbar(
        label="MNWC " +  "h " + args.parameter, orientation="horizontal"
    )

    plt.subplot(1, 4, 2)
    """
        plt.pcolormesh(
            np.asarray(lon),
            np.asarray(lat),
            diff[k],
            cmap="RdBu_r",
            vmin=-5,
            vmax=5,
        )

        """
    plt.scatter(
        obs["longitude"],
        obs["latitude"],
        s=10,
        c=diff_point,
        cmap="RdBu_r",
        vmin=(-5),
        vmax=5,
    )
        #"""
    plt.xlim(0, 35)
    plt.ylim(55, 75)
    cbar = plt.colorbar(
        label="mnwc - obs " + "h " + args.parameter, orientation="horizontal"
    )

        
    plt.subplot(1, 4, 3)
    plt.scatter(
        obs["longitude"],
        obs["latitude"],
        s=10,
        #c=obs_diff["WS1bias"], 
        c=obs_diff,
        #c=ml_fcst[k]["biasc"],
        cmap="RdBu_r",
        vmin=(-5),
        vmax=5,
        )
        #"""
    plt.xlim(0, 35)
    plt.ylim(55, 75)
    cbar = plt.colorbar(
        label="Diff to obs points " + "h " + args.parameter, orientation="horizontal"
    )
        
    plt.subplot(1, 4, 4)
    plt.pcolormesh(
        np.asarray(lon),
        np.asarray(lat),
        diff,
        cmap="RdBu_r",
        vmin=-5,
        vmax=5,
    )
    plt.xlim(0, 35)
    plt.ylim(55, 75)
    cbar = plt.colorbar(
        label="Diff " + "h " + args.parameter, orientation="horizontal"
    )

    # plt.show()
    plt.savefig("testi_" + args.parameter + ".png")


    if args.plot:
        plot(obs, background, output, diff, lons, lats, args)

if __name__ == "__main__":
    main()
