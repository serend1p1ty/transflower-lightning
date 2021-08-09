#!/bin/bash

set -ex

vis=false
if [ "$1" == "vis" ]; then
    shift 1
    vis=true
fi

EXP_NAME=$2

./inference_pipeline.sh $@

mkdir res_dir/videos/videos -p
cp inference/generated/$EXP_NAME/videos/*.bvh inference/generated/$EXP_NAME/videos/*mp4_music.mp4 res_dir
for x in res_dir/*.mp4_music.mp4; do
    base_name=$(basename $x)
    base_name=${base_name%.mp4_music.mp4}
    echo $base_name
    ffmpeg -i $x res_dir/$base_name.wav
    mv $x res_dir/videos/videos/"$base_name"_skeleton.mp4
done

# beat alignment metrics
if $vis; then
    python my_tools/eval_beat_metrics.py --vis
else
    python my_tools/eval_beat_metrics.py
fi

# FID metrics
mkdir fid_data/$EXP_NAME -p
cp inference/generated/$EXP_NAME/predicted_mods/*expmap_scaled_20.generated.npy fid_data/$EXP_NAME
./feature_extraction/get_motion_scalers.sh fid_data $EXP_NAME
python analysis/sandbox_fid.py --expname $EXP_NAME
