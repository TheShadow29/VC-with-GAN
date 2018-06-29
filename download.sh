#!/usr/bin/env bash
#mkdir -p dataset/vcc2016/wav
#cd dataset
#wget "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2042/vcc2016_training.zip"
#wget "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2042/evaluation_all.zip"
#unzip vcc2016_training.zip
#mv vcc2016_training "./vcc2016/wav/Training Set"
#unzip evaluation_all.zip -d "vcc2016/wav/Testing Set"
#rm evaluation_all.zip vcc2016_training.zip
#cd ..

mkdir -p dataset/vcc2018/wav
cd dataset
wget "http://136.187.108.4/vcc2018/vcc2018_training.zip"
unzip vcc2018_training.zip
mv vcc2018_training "./vcc2018/wav/Training Set"
rm vcc2018_training.zip
cd ..
