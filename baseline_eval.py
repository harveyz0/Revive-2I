import os
from os.path import join, exists
import csv
from classification.classifier import classify
from shutil import copyfile, rmtree

from eval.scores import main

script_home = os.path.dirname(__file__)

output_directory = os.path.join(script_home, "outputs/txt-guid-i2i-samples")


def move_all_images(output_directory=output_directory):
    all_dogs = join(output_directory, "all_dogs")
    if exists(all_dogs):
        rmtree(all_dogs)
    os.makedirs(all_dogs, exist_ok=True)
    for dir in os.listdir(output_directory):
        if dir == "all_dogs":
            continue
        for f in os.listdir(join(output_directory, dir)):
            copyfile(join(output_directory, dir, f), join(all_dogs, f))


def run_classifier(output_directory=output_directory):
    move_all_images(output_directory)
    if not os.path.exists(output_directory):
        print(f"ERROR : {output_directory} does not exist")
        raise FileNotFoundError(f"{output_directory} does not exist")
    all_outs = []
    for d in os.listdir(output_directory):
        print(f"Processing {d}")
        csv_name = d.split("_", 7)[-1] + ".csv"
        all_outs += classify(os.path.join(output_directory, d),
                             os.path.join(output_directory, csv_name),
                             "data/imagenet_classes.txt")
    with open(os.path.join("outputs", "allthem.csv"), 'w') as out_file:
        c = csv.DictWriter(out_file,
                           fieldnames=all_outs[0].keys(),
                           dialect='unix')
        c.writeheader()
        c.writerows(all_outs)


def get_scores():
    main(join(output_directory, "all_dogs"), 'data/skull2dog/testB')


run_classifier()
get_scores()
