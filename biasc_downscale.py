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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--wg_data", action="store", type=str, required=True)
    parser.add_argument("--ppa_data", action="store", type=str, required=True)
    parser.add_argument("--ws_data", action="store", type=str, required=True)
    parser.add_argument("--rh_data", action="store", type=str, required=True)
    parser.add_argument("--t2_data", action="store", type=str, required=True)
    parser.add_argument("--wd_data", action="store", type=str, required=True)
    parser.add_argument("--q2_data", action="store", type=str, required=True)
    parser.add_argument("--nl_data", action="store", type=str, required=True)
    parser.add_argument("--dem_data", action="store", type=str, default="DEM_100m-Int16.tif")
    parser.add_argument("--dem_downs", action="store", type=str, default="elev_100m_1000m.nc")
    parser.add_argument("--lsm_downs", action="store", type=str, default="maa_meri_lcc_1000.nc")
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)

    args = parser.parse_args()

    allowed_params = ["temperature", "humidity", "windspeed", "gust"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)

    return args

def read_grid(args):
    """Top function to read "all" gridded data"""
    # Define the grib-file used as background/"parameter_data"
    if args.parameter == "temperature":
        parameter_data = args.t2_data
    elif args.parameter == "windspeed":
        parameter_data = args.ws_data
    elif args.parameter == "gust":
        parameter_data = args.wg_data
    elif args.parameter == "humidity":
        parameter_data = args.rh_data

    lons, lats, vals, analysistime, forecasttime = read_grib(parameter_data, True)

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


def read_ml_grid(args):
    _, _, ws, _, _ = read_grib(args.ws_data, False)
    _, _, rh, _, _ = read_grib(args.rh_data, False)
    _, _, q2, _, _ = read_grib(args.q2_data, False)
    _, _, cl, _, _ = read_grib(args.nl_data, False)
    _, _, ps, _, _ = read_grib(args.ppa_data, False)
    _, _, t2, _, _ = read_grib(args.t2_data, False)
    _, _, wg, _, _ = read_grib(args.wg_data, False)
    _, _, wd, _, _ = read_grib(args.wd_data, False)

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
    ps = ps / 100
    cl = np.around(cl / 0.125, 0)

    return ws, rh, t2, wg, cl, ps, wd, q2

def interpolate_single_time(
    grid,
    background,
    points,
    obs,
    obs_to_background_variance_ratio,
    pobs,
    structure,
    max_points,
    idx,
    q,
):
    # perform optimal interpolation
    tmp_output = gridpp.optimal_interpolation(
        grid,
        background,
        points,
        obs[idx]["biasc"].to_numpy(),
        obs_to_background_variance_ratio,
        pobs,
        structure,
        max_points,
    )

    print(
        "step {} min grid: {:.1f} max grid: {:.1f}".format(
            idx, np.amin(tmp_output), np.amax(tmp_output)
        )
    )

    if q is not None:
        # return index and output, so that the results can
        # later be sorted correctly
        q.put((idx, tmp_output))
    else:
        return tmp_output


def interpolate(grid, points, background, obs, args, lc):
    """Perform optimal interpolation"""

    output = []

    # create a mask to restrict the modifications only to land area (where lc = 1)
    lc0 = np.logical_not(lc).astype(int)

    # Interpolate background data to observation points
    # When bias is gridded then background is zero so pobs is just array of zeros
    pobs = gridpp.nearest(grid, points, background)

    # For 1km resolution 
    # Barnes structure function with horizontal decorrelation length 30km,
    # vertical decorrelation length 500m
    structure = gridpp.BarnesStructure(30000, 500, 0.5) #(30000, 200, 0.5)

    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 20

    # error variance ratio between observations and background
    # smaller values -> more trust to observations
    obs_to_background_variance_ratio = np.full(points.size(), 0.01)

    if args.disable_multiprocessing:
        output = [
            interpolate_single_time(
                grid,
                background,
                points,
                obs,
                obs_to_background_variance_ratio,
                pobs,
                structure,
                max_points,
                x,
                None,
            )
            for x in range(len(obs))
        ]

    else:
        q = Queue()
        processes = []
        outputd = {}

        for i in range(len(obs)):
            processes.append(
                Process(
                    target=interpolate_single_time,
                    args=(
                        grid,
                        background,
                        points,
                        obs,
                        obs_to_background_variance_ratio,
                        pobs,
                        structure,
                        max_points,
                        i,
                        q,
                    ),
                )
            )
            processes[-1].start()

        for p in processes:
            # get return values from queue
            # they might be in any order (non-consecutive)
            ret = q.get()
            outputd[ret[0]] = ret[1]

        for p in processes:
            p.join()

        for i in range(len(obs)):
            # sort return values from 0 to 8
            output.append(outputd[i])

    return output


def train_data(
    args, points, obs, grid, topo, ws, rh, t2, wg, cl, ps, wd, q2, forecasttime
):
    data = pd.DataFrame()
    tmp_data = pd.DataFrame()

    # Calculate the bias at the leadtime 1h which is used in ML model and for the obs analysis
    if args.parameter == "temperature":
        forecast = gridpp.nearest(grid, points, t2[1])
    elif args.parameter == "humidity":
        forecast = gridpp.nearest(grid, points, rh[1])
    elif args.parameter == "windspeed":
        forecast = gridpp.nearest(grid, points, ws[1])
    elif args.parameter == "gust":
        forecast = gridpp.nearest(grid, points, wg[1])
        # print("havainto:", obs.iloc[:, 5])
    bias = forecast - obs.iloc[:, 5]

    # Calculate the elevation difference between the model and the obs points
    model_elev = gridpp.nearest(grid, points, topo)
    ElevD = model_elev - obs["elevation"]
    obs_elevation = obs["elevation"]

    # MNWC analystime 0h is not used at all, start from forecasttime 1 (used as observation analysis)
    for j in range(1, len(forecasttime)):
        tmp_data["bias"] = bias
        tmp_data["obs_lat"] = obs["latitude"]
        tmp_data["obs_lon"] = obs["longitude"]
        tmp_data["ElevD"] = ElevD
        tmp_data["obs_elevation"] = obs_elevation
        tmp_data["S10M"] = gridpp.nearest(grid, points, ws[j])
        tmp_data["D10M"] = gridpp.nearest(grid, points, wd[j])
        tmp_data["GMAX"] = gridpp.nearest(grid, points, wg[j])
        tmp_data["PMSL"] = gridpp.nearest(grid, points, ps[j])
        tmp_data["CCLOW"] = gridpp.nearest(grid, points, cl[j])
        tmp_data["RH2M"] = gridpp.nearest(grid, points, rh[j])
        tmp_data["T2M"] = gridpp.nearest(grid, points, t2[j])
        tmp_data["Q2M"] = gridpp.nearest(grid, points, q2[j])
        tmp_data["leadtime"] = j
        tmp_data["forecasttime"] = forecasttime[j]
        data = data.append(tmp_data, ignore_index=True)
        # data = pd.concat([data,tmp_data],join="inner")
        # print("tmp:",tmp_points)
    return data


# Modify time to sin/cos representation
def encode(data, col, max_val):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    return data


# Modify dataframe for xgboost model according parameter
def modify(data, param):
    data = data.assign(month=data.forecasttime.dt.month)
    data = data.assign(hour=data.forecasttime.dt.hour)
    data = encode(data, "month", 12)
    data = encode(data, "hour", 24)
    data = data.drop(["month", "hour"], axis=1)
    # data = data.dropna()
    if param == "temperature":
        data = data.drop(["GMAX", "D10M"], axis=1)
        data = data[
            [
                "leadtime",
                "T2M",
                "S10M",
                "RH2M",
                "PMSL",
                "Q2M",
                "CCLOW",
                "obs_lat",
                "obs_lon",
                "ElevD",
                "obs_elevation",
                "bias",
                "month_sin",
                "month_cos",
                "hour_sin",
                "hour_cos",
            ]
        ]
        data.rename(columns={"bias": "T1bias"}, inplace=True)
    elif param == "windspeed":
        data = data.drop(["RH2M", "Q2M"], axis=1)
        data = data[
            [
                "leadtime",
                "D10M",
                "T2M",
                "S10M",
                "PMSL",
                "CCLOW",
                "GMAX",
                "obs_lat",
                "obs_lon",
                "ElevD",
                "obs_elevation",
                "bias",
                "month_sin",
                "month_cos",
                "hour_sin",
                "hour_cos",
            ]
        ]
        data.rename(columns={"bias": "WS1bias"}, inplace=True)
    elif param == "gust":
        data = data.drop(["Q2M", "CCLOW"], axis=1)
        data = data[
            [
                "leadtime",
                "D10M",
                "T2M",
                "S10M",
                "RH2M",
                "PMSL",
                "GMAX",
                "obs_lat",
                "obs_lon",
                "ElevD",
                "obs_elevation",
                "bias",
                "month_sin",
                "month_cos",
                "hour_sin",
                "hour_cos",
            ]
        ]
        data.rename(columns={"bias": "WG1bias"}, inplace=True)
    elif param == "humidity":
        data = data.drop(["GMAX", "D10M"], axis=1)
        data = data[
            [
                "leadtime",
                "T2M",
                "S10M",
                "RH2M",
                "PMSL",
                "Q2M",
                "CCLOW",
                "obs_lat",
                "obs_lon",
                "ElevD",
                "obs_elevation",
                "bias",
                "month_sin",
                "month_cos",
                "hour_sin",
                "hour_cos",
            ]
        ]
        data.rename(columns={"bias": "RH1bias"}, inplace=True)

    # modify data precision from f64 to f32
    data[data.select_dtypes(np.float64).columns] = data.select_dtypes(
        np.float64
    ).astype(np.float32)

    return data


# Calculate ml model point forecast
def ml_forecast(ml_data, param):
    # Load model from S3, I'm unable to make it happen tough...
    if param == "temperature":
        mlname = "T2m"
    elif param == "windspeed":
        mlname = "WS"
    elif param == "gust":
        mlname = "WG"
    elif param == "humidity":
        mlname = "RH"

    regressor = joblib.load("xgb_" + mlname + "_1023.joblib")

    # Check that you have all the leadtimes (0-9)
    ajat = sorted(ml_data["leadtime"].unique().tolist())

    # Leadtime 1 == obs analysis --> ML model not run for this
    lt1 = ml_data[ml_data["leadtime"] == 1]
    ml_data = ml_data[ml_data.leadtime != 1]

    # Calculate xgboost based bias correction for the leadtimes 2...x
    ml_data["biasc"] = regressor.predict(ml_data)

    if param == "temperature":
        lt1["biasc"] = lt1["T1bias"]
    elif param == "windspeed":
        lt1["biasc"] = lt1["WS1bias"]
    elif param == "gust":
        lt1["biasc"] = lt1["WG1bias"]
    elif param == "humidity":
        lt1["biasc"] = lt1["RH1bias"]

    # Store data to list where each leadtime is it's own list, first leadtime is the obs analysis
    ml_results = []
    ml_results.append(lt1)

    # Store bias correected forecast for leadtimes 2...x to list
    for j in range(1, len(ajat)):
        lt_tmp = ml_data[ml_data["leadtime"] == j + 1]
        ml_results.append(lt_tmp)

    return ml_results


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

    ws, rh, t2, wg, cl, ps, wd, q2 = read_ml_grid(args)
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
    # prepare dataframe for ML code, pandas df
    data = train_data(
        args, points, obs, grid, topo, ws, rh, t2, wg, cl, ps, wd, q2, forecasttime
    )

    # preprocess data
    ml_data = modify(data, args.parameter)

    # Produce ML forecast for all the leadtimes
    ml_fcst = ml_forecast(ml_data, args.parameter)
    mlt = time.time()
    timedif = mlt - ot
    print("Producing ML forecasts takes:", round(timedif, 1), "seconds")
    #####
    # use 1km dem and lsm in gridding!
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

    ###LSM in 1km 0/1
    lc_file = nc.Dataset(args.lsm_downs, 'r')
    # Access latitude, longitude, and elevation data
    latlc = lc_file.variables['y'][:]
    lonlc = lc_file.variables['x'][:]
    lc_val = lc_file.variables['Band1'][:]
      
    lc_file.close()
    latlc = np.array(latlc)
    lonlc = np.array(lonlc)
    lc_val = np.array(lc_val)

    #lcdem = np.zeros_like(elev)
    # lat & lon are 1-d when elev is 2d, modify to 2d
    lonlc, latlc = np.meshgrid(lonlc, latlc)
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
    lonlc, latlc = transform(lcc_proj, target_proj, lonlc, latlc)

    # create a grid class needed in gridpp
    griddem = gridpp.Grid(lat, lon , elev , lc_val)
    
    # do the downscaling to 1km
    if args.parameter == "temperature":
        # choose a constant gradient 
        gradient = -0.0065 # C/m
        #gradient = -0.65 # for grazy testing
        grid1km = gridpp.simple_gradient(grid, griddem, background, gradient)
    else:
        # nearest also an option!
        grid1km = gridpp.bilinear(grid, griddem, background)
    
    grid1km0 = copy.copy(grid1km)
    grid1km0[grid1km0 != 0] = 0
    background = grid1km
    background0 = grid1km0

    # Interpolate ML point forecasts for bias correction + 0h analysis time
    # Add lsm info to obs points from 1km lsm mask since obs don't have that info
    lsm_point = gridpp.nearest(griddem, points, lc_val)
    points = gridpp.Points(
        obs["latitude"].to_numpy(),
        obs["longitude"].to_numpy(),
        obs["elevation"].to_numpy(),
        lsm_point,
    )
    diff = interpolate(griddem, points, background0[0], ml_fcst, args, lcdem)
    oit = time.time()
    timedif = oit - mlt
    print("Interpolating forecasts takes:", round(timedif, 1), "seconds")
    # calculate the final bias corrected forecast fields: MNWC - bias_correction
    # and convert parameter to T-K or RH-0TO1
    output = []
    for j in range(0, len(diff)):
        tmp_output = background[j + 1] - diff[j]
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
        output.append(tmp_output)

    # Remove analysistime (leadtime=0), because correction is not made for that time
    forecasttime.pop(0)
    assert len(forecasttime) == len(output)
    # check for missing data in output
    if np.isnan(output).any() or np.any(output == None):
        print("Bias correction output contains NaN/None values")
        # replace nan/None with missing data in grib 9999
        output = np.where(np.isnan(output) | (output == None), 9999, output)
        # exit()
    write_grib(args, analysistime, forecasttime, output)

    obs_diff=ml_data[ml_data["leadtime"] == 1]
    

    # plot the results
    vmin = np.amin(background_orig)
    vmax = np.amax(background_orig)
    print("vmin:", vmin)
    print("vmax:", vmax)

    vmin1 =  np.amin(output)
    vmax1 =  np.amax(output)
    print("vmin:", vmin)
    print("vmax:", vmax)

    for k in range(0, len(output)):
        fc_point = gridpp.nearest(griddem, points, diff[k])
        plt.figure(figsize=(17, 6), dpi=80)
        plt.subplot(1, 4, 1)
        plt.pcolormesh(
            np.asarray(lon),
            np.asarray(lat),
            background[k],
            cmap="Spectral_r",  # "RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="MNWC " + str(k) + "h " + args.parameter, orientation="horizontal"
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
        ml_fcst[k]["obs_lon"],
        ml_fcst[k]["obs_lat"],
        #obs_diff["obs_lon"],
        #obs_diff["obs_lat"],
        s=10,
        #c=obs_diff["WS1bias"], 
        #c=fc_point,
        c=ml_fcst[k]["biasc"],
        cmap="RdBu_r",
        vmin=(-5),
        vmax=5,
        )
        #"""
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="ml_fcst_point " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        
        plt.subplot(1, 4, 3)
        plt.scatter(
        obs_diff["obs_lon"],
        obs_diff["obs_lat"],
        s=10,
        #c=obs_diff["WS1bias"], 
        c=fc_point,
        #c=ml_fcst[k]["biasc"],
        cmap="RdBu_r",
        vmin=(-5),
        vmax=5,
        )
        #"""
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="Diff_to_points " + str(k) + "h " + args.parameter, orientation="horizontal"
        )
        
        plt.subplot(1, 4, 4)
        plt.pcolormesh(
            np.asarray(lon),
            np.asarray(lat),
            diff[k],
            cmap="RdBu_r",
            vmin=-5,
            vmax=5,
        )
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="Diff " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        # plt.show()
        plt.savefig("testi1_" + args.parameter + str(k) + ".png")

        plt.figure(2)
        plt.figure(figsize=(14, 6), dpi=80)
        plt.subplot(1, 3, 1)
        plt.pcolormesh(
            np.asarray(lon),
            np.asarray(lat),
            background[k],
            cmap="Spectral_r",  # "RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(23,25)#(0, 35)
        plt.ylim(67.5,68.5)#(55, 75)
        cbar = plt.colorbar(
            label="MNWC " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        plt.subplot(1, 3, 2)
        plt.pcolormesh(
            np.asarray(lon),
            np.asarray(lat),
            output[k],
            cmap="Spectral_r",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlim(22,25)#(0, 35)
        plt.ylim(67.5,68.5)#(55, 75)
        cbar = plt.colorbar(
            label="Output " + str(k) + "h " + args.parameter, orientation="horizontal"
        )
        plt.subplot(1, 3, 3)
        plt.pcolormesh(
            np.asarray(lon),
            np.asarray(lat),
            diff[k],
            cmap="RdBu_r",
            vmin=(-5),
            vmax=5,
        )
        plt.xlim(22,25)#(0, 35)
        plt.ylim(67.5,68.5)#(55, 75)
        cbar = plt.colorbar(
            label="Diff " + str(k) + "h " + args.parameter, orientation="horizontal"
        )
        plt.savefig("output_" + args.parameter + str(k) + ".png")


    if args.plot:
        plot(obs, background, output, diff, lons, lats, args)

if __name__ == "__main__":
    main()
