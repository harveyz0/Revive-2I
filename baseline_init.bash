#!/bin/bash

# Stole from https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
RED='\033[01;31m'
NC='\033[01;0m' # No Color

script_home=$(dirname "$0")

pushd "$script_home" &> /dev/null


if [ "$1" = "-a" ]
then
	exit 0
fi


conda_setup() {
	local error_code=0
	if ! command -v conda > /dev/null ; then
		echo -e "${RED}ERROR : Can not find conda command ; bailing${NC}"
		error_code=2
		return $error_code
	fi

	conda env create -f ./environment.yaml
	error_code=$?

	if [ ! -d "$script_home/data/skull2dog/testA" ] ; then
		echo -e "${RED}ERROR : Missing skull2dog test data${NC}"
		echo -e "${RED}ERROR : Please unzip skull2dog data set to the path $script_home/data/skull2dog/testA${NC}"
		error_code=2
	fi
	if [ $error_code -eq 0 ]
	then
		echo "Please run conda activate harveyz_revive "
	else
		echo -e "${RED}ERROR : Something went wrong with conda create. You may have to delete environment harveyz_revive and run this script again.${NC}"
	fi

	return $error_code
}

tree_setup() {
	local model_dir="models/ldm/stable-diffusion-v1"
	local data_dir="data/skull2dog"
	local model_file="sd-v1-4-full-ema.ckpt"
	local error_code=0
	mkdir -p $model_dir
	if [ -f  $model_file ]
	then
		cp $model_file $model_dir/model.ckpt
	else
		echo -e "${RED}ERROR : Can not find model ckpt file ${model_file}${NC}"
		error_code=2
	fi


	test_zip="test.zip"
	traina_zip="trainA.zip"
	trainb_zip="trainB.zip"
	mkdir -p $data_dir
	if ! command -v unzip > /dev/null
	then
		echo -e "${RED}ERROR : Can not find the unzip command. Please install it.${NC}"
		error_code=2
	fi

	if [ -f $test_zip ]
	then
		unzip -o $test_zip -d $data_dir
	else
		echo -e "${RED}ERROR : Can not find the ${test_zip} file. Please download it from my OneDrive${NC}"
		error_code=2
	fi

	if [ -f $traina_zip ]
	then
		unzip -o $traina_zip -d $data_dir
	else
		echo -e "${RED}ERROR : Can not find the ${traina_zip} file. Please download it from my OneDrive${NC}"
		error_code=2
	fi

	if [ -f $trainb_zip ]
	then
		unzip -o $trainb_zip -d $data_dir
	else
		echo -e "${RED}ERROR : Can not find the ${trainb_zip} file. Please download it from my OneDrive${NC}"
		error_code=2
	fi

	return $error_code
}
tree_setup
err=$?

if [ $err -eq 0 ]
then
	conda_setup
	err=$?
fi


popd &> /dev/null

exit "$err"
