#!/bin/bash

fps=15

now=$(date +'%Y%m%d_%H%M%S')


ffmpeg -r $fps -f image2 -s 960x720 -pattern_type glob -i 'run_details/iter_mask_*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/masks_${now}.mp4
ffmpeg -r $fps -f image2 -s 960x720 -pattern_type glob -i 'run_details/iter_weight_*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/weight_${now}.mp4
ffmpeg -r $fps -f image2 -s 960x720 -pattern_type glob -i 'run_details/iter_pruner_*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/pruner_${now}.mp4
# rm run_details/*
