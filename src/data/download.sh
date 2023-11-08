#!/usr/bin/env bash

# Check path. If we are not in root directory, then go to root directory
while [ ! -d "src" ]; do
    cd ..
done

mkdir -p data/raw/
cd data/raw/

wget https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip
unzip filtered_paranmt.zip
rm filtered_paranmt.zip
