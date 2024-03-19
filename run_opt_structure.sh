#!/usr/bin/python3.9
# Script to run ML realtime forecasts for testing

#python3.9 --version

#cd /home/users/hietal/statcal/python_projects/snwc_bc/

parameter=$1 # T2m, RH, WS or WG

WEEKDAY=`date +"%a"`
HOD=`date +"%H"`
AIKA1=`date "+%Y%m%d%H"  -u`
HH=2
NN=$(($AIKA1-$HH))
bucket="s3://routines-data/mnwc-biascorrection/production/"
echo $NN

#export TMPDIR=/data/hietal/testi

if [ "$parameter" == "T2m" ]; then
  pyparam="temperature"
  datafile="$bucket""$NN"00/T-K.grib2
  s3cmd get $datafile
elif [ "$parameter" == "RH" ]; then
  pyparam="humidity"
  datafile="$bucket""$NN"00/RH-0TO1.grib2
  s3cmd get $datafile
elif [ "$parameter" == "WS" ]; then
  pyparam="windspeed"
  datafile="$bucket""$NN"00/FF-MS.grib2
  s3cmd get $datafile 
elif [ "$parameter" == "WG" ]; then
  pyparam="gust"
  datafile="$bucket""$NN"00/FFG-MS.grib2
  s3cmd get $datafile #/data/hietal/testi/FFG-MS.grib2
else
  echo "parameter must be T2m, RH, WS or WG"
  exit 1
fi

python3 optimize_structure.py --topography_data "$bucket""$NN"00/Z-M2S2.grib2 --landseacover "$bucket""$NN"00/LC-0TO1.grib2 --parameter_data $datafile --output testi_"$parameter".grib2 --parameter "$pyparam"

#rm -r /data/hietal/testi/tmp*
