#!/bin/bash

fps=15

rm videos/*

ffmpeg -r $fps -f image2 -s 960x720 -pattern_type glob -i 'run_details/iter_mask_*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/masks.mp4
ffmpeg -r $fps -f image2 -s 960x720 -pattern_type glob -i 'run_details/iter_weight_*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/weight.mp4
ffmpeg -r $fps -f image2 -s 960x720 -pattern_type glob -i 'run_details/iter_pruner_*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/pruner.mp4
