#!/bin/bash

set -eu

mkdir $1
gsutil -m cp -r gs://metagen/data/dance_combined3/*expmap_cr_scaled_20.npy gs://metagen/data/dance_combined3/*audio_feats_scaled_20.npy gs://metagen/data/dance_combined3/*pkl gs://metagen/data/dance_combined3/*sav gs://metagen/data/dance_combined3/base_filenames* $1
gsutil -m cp -r gs://metagen/data/dance_combined3/*dance_style.npy $1