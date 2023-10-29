* Notes
There are a lot of hard coded paths around so you're probably best running this from the root of the repo.

* Setup
Download all the files from my OneDrive to the root of this repo. Once you've done that run baseline_init.bash. This should put all the images and models into the correct places. Once that script is done make sure you activate the conda environment harveyz_revive.

* Train
baseline_train.py should be run next. This will take a while, like about 162.40 minutes.

* Evaluate
baseline_eval.py will run the classifier and then score the results. This should only take a few minutes. It'll print the scores out to stdout and to outputs/scores.txt 
