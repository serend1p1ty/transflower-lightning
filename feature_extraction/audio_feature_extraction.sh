#!/bin/bash

folder=$1
py=python3
n=$(nproc)
#n=6

#find $1 -type f -iname "*.mp3" -exec basename \{\} .mp3 \; > $1/base_filenames.txt

fps=20
format=wav

###SEQUENCE TO PROCESS DATA WHEN NEEDING TO COMPUTE NORMALIZATION TRANSFORMS
# [80, 3, 100t], x.wav_multi_mel_80.npy
mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names multi_mel --mel_feature_size 80 --fps 100 # fps=100 coz thats what ddc expects
# [20t, 512], x.wav_multi_mel_80.npy_ddc_hidden.npy
mpirun -n $n $py ./feature_extraction/generate_ddc_features.py $@ --audio_format $format --experiment_name block_placement_ddc2 --checkpoint 130000 --checkpoints_dir feature_extraction --fps $fps
# concat([20t, 80], [20t, 1], [20t, 2]) = [20t, 83], x.wav.audio_feats.npy
mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names mel,envelope,madmombeats --mel_feature_size 80 --fps $fps --combined_feature_name audio_feats
# concatenate all ddc_hidden features [N, 512], then fit and save pca transforms, wav_multi_mel_80.npy_ddc_hidden_pca_transform.pkl
mpirun -n 1 $py ./feature_extraction/extract_transform.py $1 --feature_name ${format}_multi_mel_80.npy_ddc_hidden --transforms pca_transform
# apply pca: [20t, 80] -> [20t, 2], x.ddcpca.npy, ddcpca_scaler.pkl (transform file, which is named by scaler)
mpirun -n $n $py ./feature_extraction/apply_transforms.py $@ --feature_name ${format}_multi_mel_80.npy_ddc_hidden --transform_name pca_transform --pca_dims 2 --new_feature_name ddcpca
# save filenames
./feature_extraction/script_to_list_filenames $1 $format
# concat([20t, 83], [20t, 2]) = [20t, 85], x.feats_ddcpca.npy
mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names ${format}_audio_feats,ddcpca --new_feature_name feats_ddcpca
# save StandardScaler transform, feats_ddcpca_scaler.pkl
mpirun -n 1 $py ./feature_extraction/extract_transform2.py $1 --feature_name feats_ddcpca --transforms scaler
# apply StandardScaler transform, x.audio_feats_scaled_20.npy, audio_feats_scaled_20_scaler.pkl (transform file)
mpirun -n $n $py ./feature_extraction/apply_transforms.py $@ --feature_name feats_ddcpca --transform_name scaler --new_feature_name audio_feats_scaled_${fps}
##feature_extraction/duplicate_features.sh data/dance_combined3 audio_feats_scaled_20

###SEQUENCE WHEN USING EXISTING TRANSFORMS (SO NO NEED T RECOMPUTE THEM)
#mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names multi_mel --mel_feature_size 80 --fps 100 # fps=100 coz thats what ddc expects
#mpirun -n $n $py ./feature_extraction/generate_ddc_features.py $@ --audio_format $format --experiment_name block_placement_ddc2 --checkpoint 130000 --checkpoints_dir feature_extraction --fps $fps
#mpirun -n $n $py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names mel,envelope,madmombeats --mel_feature_size 80 --fps $fps --combined_feature_name audio_feats
#mpirun -n $n $py ./feature_extraction/apply_transforms.py $@ --feature_name ${format}_multi_mel_80.npy_ddc_hidden --transform_name pca_transform --pca_dims 2 --new_feature_name ddcpca
#./feature_extraction/script_to_list_filenames $1 $format
#mpirun -n $n $py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names ${format}_audio_feats,ddcpca --new_feature_name feats_ddcpca
#mpirun -n $n $py ./feature_extraction/apply_transforms.py $@ --feature_name feats_ddcpca --transform_name scaler --new_feature_name audio_feats_scaled_${fps}

###NOMPI
#$py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names multi_mel --mel_feature_size 80 --fps 100 # fps=100 coz thats what ddc expects
#$py ./feature_extraction/generate_ddc_features.py $@ --audio_format $format --experiment_name block_placement_ddc2 --checkpoint 130000 --checkpoints_dir feature_extraction --fps $fps
#$py ./feature_extraction/process_audio.py $@ --audio_format $format --feature_names mel,envelope,madmombeats --mel_feature_size 80 --fps $fps --combined_feature_name audio_feats
#$py ./feature_extraction/apply_transforms.py $@ --feature_name ${format}_multi_mel_80.npy_ddc_hidden --transform_name pca_transform --pca_dims 2 --new_feature_name ddcpca
#./feature_extraction/script_to_list_filenames $1 $format
#$py ./feature_extraction/combine_feats.py $@ $1/base_filenames.txt --feature_names ${format}_audio_feats,ddcpca --new_feature_name feats_ddcpca
#$py ./feature_extraction/apply_transforms.py $@ --feature_name feats_ddcpca --transform_name scaler --new_feature_name audio_feats_scaled_${fps}
