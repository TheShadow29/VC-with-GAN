#!/usr/bin/env bash
mkdir -p dataset/vcc2016/wav
cd dataset
wget "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2042/vcc2016_training.zip"
wget "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2042/evaluation_all.zip"
unzip vcc2016_training.zip
mv vcc2016_training "./vcc2016/wav/Training Set"
unzip evaluation_all.zip -d "vcc2016/wav/Testing Set"
rm evaluation_all.zip vcc2016_training.zip
cd ..
