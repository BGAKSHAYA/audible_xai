"""
Code extracted from https://github.com/albermax/innvestigate/tree/master/src/innvestigate/utils
Alber, M., Lapuschkin, S., Seegerer, P., Hägele, M., Schütt, K. T., Montavon, G., Samek, W., Müller, K. R., Dähne, S., & Kindermans, P. J. (2019).
iNNvestigate neural networks! Journal of Machine Learning Research, 20.
"""


import numpy as np
import logging
import innvestigate_utils.visualizations as ivis
logger = logging.getLogger('audible_xai')

try:
    import innvestigate
    import innvestigate.utils as iutils

except (ModuleNotFoundError, AttributeError):
    logger.warning("Skipped loading innvestigate modules, Only TFv2 supported saliency methods will be executed")
    pass


def preprocess(X, net):
    X = X.copy()
    X = net["preprocess_f"](X)
    return X


def postprocess(X, color_conversion, channels_first):
    X = X.copy()
    X = iutils.postprocess_images(
        X, color_coding=color_conversion, channels_first=channels_first
    )
    return X


def image(X):
    X = X.copy()
    return ivis.project(X, absmax=255.0, input_is_positive_only=True)


def bk_proj(X):
    X = ivis.clip_quantile(X, 1)
    return ivis.project(X)


def heatmap(X):
    # X = ivis.gamma(X, minamp=0, gamma=0.95)
    return ivis.heatmap(X)


def graymap(X):
    return ivis.graymap(np.abs(X), input_is_positive_only=True)
