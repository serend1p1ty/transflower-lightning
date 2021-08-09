#!/bin/bash

set -e
set -x

SONGS_DIR=$1
EXP_NAME=$2
SEED_STYLE=$3
SEED_DIR=$4
ALL_STYLES=("free_style" "casual" "hiphop" "break_dance" "groovenet" "casual2" "random")

if [[ ! " ${ALL_STYLES[@]} " =~ " $SEED_STYLE " ]]; then
    echo "Valid seed style: free_style, casual, hiphop, break_dance, groovenet, casual2, random"
    exit
fi

# copy seeds and scalers to song directory
cp $SEED_DIR/* $SONGS_DIR

# create the seed motion for each audio
for audio in $(ls $SONGS_DIR/*.wav | xargs -n 1 -i basename {} .wav); do
    if [ $SEED_STYLE == "random" ]; then
        rand_id=$RANDOM%6
        cp $SONGS_DIR/seed_${ALL_STYLES[$rand_id]}.npy $SONGS_DIR/$audio.expmap_scaled_20.npy
    else
        cp $SONGS_DIR/seed_$SEED_STYLE.npy $SONGS_DIR/$audio.expmap_scaled_20.npy
    fi
done

# extract audio features
chmod +x ./feature_extraction/audio_feature_extraction_test.sh
chmod +x ./feature_extraction/script_to_list_filenames
./feature_extraction/audio_feature_extraction_test.sh "$SONGS_DIR"

for audio_basename in $(ls $SONGS_DIR/*.wav | xargs -n 1 -i basename {} .wav); do
    # generate dance
    ./script_generate.sh $EXP_NAME $audio_basename --generate_bvh --generate_video --data_dir=$SONGS_DIR

    if [ "$5" == "vis" ]; then
        # visualization
        gsutil -m cp inference/generated/$EXP_NAME/videos/$audio_basename.bvh gs://metagendance/
        cp inference/generated/$EXP_NAME/videos/$audio_basename.mp4.mp3 inference/generated/$EXP_NAME/videos/$audio_basename.bvh.mp3
        gsutil -m cp inference/generated/$EXP_NAME/videos/$audio_basename.bvh.mp3 gs://metagendance/
        echo "Open https://guillefix.github.io/bvh_visualizer/?name=$audio_basename to see the visualization..."
    fi
done

# audio_basename="aistpp_gJB_sBM_cAll_d07_mJB2_ch09"
# # generate dance
# ./script_generate.sh $EXP_NAME $audio_basename --generate_bvh --generate_video --data_dir=$SONGS_DIR

# if [ "$5" == "vis" ]; then
#     # visualization
#     gsutil -m cp inference/generated/$EXP_NAME/videos/$audio_basename.bvh gs://metagendance/
#     cp inference/generated/$EXP_NAME/videos/$audio_basename.mp4.mp3 inference/generated/$EXP_NAME/videos/$audio_basename.bvh.mp3
#     gsutil -m cp inference/generated/$EXP_NAME/videos/$audio_basename.bvh.mp3 gs://metagendance/
#     echo "Open https://guillefix.github.io/bvh_visualizer/?name=$audio_basename to see the visualization..."
# fi
