# -*- coding: utf-8 -*-
import os
from constants.dataset_constants import DATASETS, DATASET_TYPES
import glob
import json
import logging

logger = logging.getLogger("audible_xai")

# Code adapted from https://github.com/soerenab/AudioMNIST/blob/master/preprocess_data.py
# Becker et al. Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals

def create_splits(src, dst):

    """
    Creation of text files specifying which files training, validation and test
    set consist of for each cross-validation split. 

    Parameters:
    -----------
        src: string
            Path to directory containing the directories for each subject that
            hold the preprocessed data in hdf5 format.
        dst: string
            Destination where to store the text files specifying training, 
            validation and test splits.

    """

    logger.debug("creating splits")
    splits={"digit": {   "train":[   set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2, \
                                          8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),

                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, \
                                         10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),

                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41, \
                                          4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),

                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42, \
                                          5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),

                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1, \
                                          6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54])],

                        "validate":[set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55])],

                        "test":[    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50])]},

            "gender":{  "train":[   set([36, 47, 56, 26, 12, 57, 2, 44, 50, 25, 37, 45]),
                                    set([26, 12, 57, 43, 28, 52, 25, 37, 45, 48, 53, 41]),
                                    set([43, 28, 52, 58, 59, 60, 48, 53, 41, 7, 23, 38]),
                                    set([58, 59, 60, 36, 47, 56, 7, 23, 38, 2, 44, 50])],

                        "validate":[set([43, 28, 52, 48, 53, 41]),
                                    set([58, 59, 60, 7, 23, 38]),
                                    set([36, 47, 56, 2, 44, 50]),
                                    set([26, 12, 57, 25, 37, 45])],

                        "test":[    set([58, 59, 60, 7, 23, 38]),
                                    set([36, 47, 56, 2, 44, 50]),
                                    set([26, 12, 57, 25, 37, 45]),
                                    set([43, 28, 52, 48, 53, 41])]}}

    for split in range(5):
        for modus in ["train", "validate", "test"]:
            for task in ["digit", "gender"]:
                if task == "gender" and split > 3:
                    continue
                with open(os.path.join(dst, "AlexNet_{}_{}_{}.txt".format(task, split, modus)), mode = "w") as txt_file:
                    for vp in splits[task][modus][split]:
                        for filepath in glob.glob(os.path.join(src, "{:02d}".format(vp), "*.hdf5")):
                            txt_file.write(filepath+"\n")

                with open(os.path.join(dst, "AudioNet_{}_{}_{}.txt".format(task, split, modus)), mode = "w") as txt_file:
                    for vp in splits[task][modus][split]:
                        for filepath in glob.glob(os.path.join(src, "{:02d}".format(vp), "*.hdf5")):
                            txt_file.write(filepath+"\n")


if __name__ == "__main__":
    path_configuration = json.load(open("../path_configuration.json"))
    config = path_configuration[DATASETS.AUDIO_MNIST_DIGITS.value][DATASET_TYPES.SPECTROGRAM.value]
    source = config["preprocessing"]["source"]
    destination = config["preprocessing"]["destination"]
    meta = config["preprocessing"]["meta"]

    # create training, validation and test sets
    create_splits(src=destination, dst=destination)
