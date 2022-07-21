# -*- coding: utf-8 -*-
import os
import logging
from constants.dataset_constants import DATASETS, DATASET_TYPES
import glob
import json

logger = logging.getLogger("audible_xai")


def create_splits(src, dst):
    """
    Creation of text files specifying which files training, validation and test set consists.
    """
    logger.debug("creating splits")
    # Train-validate-splits with 75%-5%-20% data
    splits = {
        "train": ["0"*(5-len(str(i))) + str(i) for i in range(0, 75)],
        "validate": ["0"*(5-len(str(i))) + str(i) for i in range(75, 80)],
        "test": ["0"*(5-len(str(i))) + str(i) for i in range(80, 100)],
    }

    dataset_file_paths = glob.glob(os.path.join(dst, '**/*.hdf5'))
    dataset_file_paths = sorted(dataset_file_paths)

    for split, split_examples in splits.items():
        files = [file for file in dataset_file_paths if os.path.basename(file).split("_")[1] in split_examples]
        with open(os.path.join(dst, "{}.txt".format(split)), mode="w") as txt_file:
            for filepath in files:
                txt_file.write(filepath + "\n")


if __name__ == "__main__":
    path_configuration = json.load(open("../path_configuration.json"))
    config = path_configuration[DATASETS.GTZAN.value][DATASET_TYPES.MELSPECTROGRAM.value]
    source = config["preprocessing"]["source"]
    destination = config["preprocessing"]["destination"]

    # create training, validation and test sets
    create_splits(src=destination, dst=destination)
