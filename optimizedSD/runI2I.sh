#!/bin/bash

# script for running the image-to-image translation 


# input will come as runI2I.sh "a realistic photo of a dog head"
prompt=$1 # the prompt to be used for the image-to-image translation

# input folder
input=$2 # the folder containing the input images

# for every image in the input folder, run the image-to-image translation and save the output in the output folder
for file in $input/*
do
    echo $file
    python optimizedSD/optimized_img2img.py --prompt "$prompt" --init-img $file --strength 0.8 --n_iter 1 --n_samples 1 --H 512 --W 512
    # python optimizedSD/optimized_img2img.py --prompt "Austrian alps" --init-img ~/sketch-mountains-input.jpg --strength 0.8 --n_iter 2 --n_samples 5 --H 512 --W 512
done