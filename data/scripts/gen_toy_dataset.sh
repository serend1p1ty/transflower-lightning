#!/bin/bash

set -e
set -x

# original dataset directory
DATA_DIR=$1
# annotations directory
ANNO_DIR=$2
# toy dataset size
TOY_SIZE=$3
# toy dataset directory
SAVE_DIR=$4

if [ ! -d $DATA_DIR ] || [ ! -d $ANNO_DIR ]
then
    echo "$DATA_DIR does not exist!!!"
    exit
fi

[ -d $SAVE_DIR ] || mkdir $SAVE_DIR

# copy a small toy dataset
videos=$(find $DATA_DIR -name "*.mp4" | head -$TOY_SIZE)
cp $videos $SAVE_DIR

# mp4->wav
find $SAVE_DIR -type f -name '*.mp4' -print0 | parallel -0 ffmpeg -i {} {.}.wav

# copy motion files and replace "cAll"
for video in $(ls $SAVE_DIR/*mp4 | xargs -n 1 basename)
do
    name_split=(${video//_/ })
    name_split[2]="cAll"
    cAll_name=$(IFS=_ ; echo "${name_split[*]}")
    cAll_name=${cAll_name/"mp4"/"pkl"}
    orig_name=${video/"mp4"/"pkl"}
    cp $ANNO_DIR/motions/$cAll_name $SAVE_DIR/$orig_name
done

# generate bvh files
python feature_extraction/process_aistpp.py $SAVE_DIR --fps 60

# extract audio and motion features
./feature_extraction/audio_feature_extraction.sh $SAVE_DIR
./feature_extraction/motion_feature_extraction.sh $SAVE_DIR
