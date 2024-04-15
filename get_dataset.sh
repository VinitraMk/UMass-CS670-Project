#!/bin/bash
data_dir="data"
if [ ! -d "$data_dir" ]; then
    echo 'Directory does not exist! Downloading data...'
    mkdir ./$data_dir
    #echo 'Downloaded data! Unzipping to data folder'
    unzip -d ./$data_dir ./COD10K-v3.zip
    rm ./COD10K-v3.zip
else
    echo 'Directory exists!'
fi