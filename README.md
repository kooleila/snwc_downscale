## Test to grid SNWC bias correction to background with 1km resolution
Can be used to test producing obs analysis/snwc bias correction in 1km resolution instead of the original 2.5km. 

## Usage 
```
sh run_biasc_downscale.sh WS # T2m, RH or WG
OR for just observation analysis use  
sh run_lsm_gridpp.sh WS # T2m, RH or WG
IF you wish to specify the area used for plotting add lat and lon info
sh run_lsm_gridpp.sh WS --lat1 60 --lat2 61 --lon1 25 --lon2 26
```
* Heavily relies on data only available at FMI.   


