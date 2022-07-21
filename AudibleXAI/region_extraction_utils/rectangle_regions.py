import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import joblib
import math
import hdbscan
import cv2
import logging

logger = logging.getLogger("audible_xai")


def get_rectangular_regions(example, saliency, dataset_params, hp, title="", remove_silence=False, retry=0):
    saliency_copy = np.copy(saliency)
    if remove_silence:
        # Temporary hack for LIME issue which selects the whole silent region as relevant to prevent OOM errors
        silent_range_start,  silent_range_end = math.inf, -math.inf
        for index in range(saliency.shape[1]-1, -1, -1):
            if np.all(saliency[:, index] == saliency_copy[0, index]):
                silent_range_start = min(silent_range_start, index)
                silent_range_end = max(silent_range_end, index)
            else:
                break
        if silent_range_start != math.inf:
            saliency[:, range(silent_range_start, silent_range_end)] = 0

    saliency[np.where(saliency < 0)] = 0

    # Stop retrying region extraction, when threshold falls below 10 and min_cluster_size is less than 10
    if hp["threshold_percentile"] < 10 or hp["min_cluster_size"] < 10:
        return []

    THRESHOLD = np.percentile(saliency, hp["threshold_percentile"])
    saliency[np.where(saliency < THRESHOLD)] = 0

    assert saliency.shape[-1] == 1, 'Audio samples with more than 1 channel is not currently supported'

    if THRESHOLD == 0:  # Dont consider 0 relevance score
        x, y, _ = np.where(saliency > THRESHOLD)
    else:
        x, y, _ = np.where(saliency >= THRESHOLD)

    data = np.array(list(zip(x, y)))
    min_x_units = dataset_params.min_x_units

    MIN_CLUSTER_SIZE = hp["min_cluster_size"]
    MIN_SAMPLES = hp["min_samples"]
    hdbscan_clf = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)

    cluster_labels = hdbscan_clf.fit_predict(data)
    labels, counts = np.unique(cluster_labels, return_counts=True)
    removed_extremes = np.where((labels != -1))
    relevant_clusters = labels[removed_extremes]

    eliminated_rectangular_regions = []
    rectangular_regions_approximations = []

    for index, cluster in enumerate(relevant_clusters):
        cluster_idxs = np.where(cluster_labels == cluster)[0]
        region_mask = np.zeros(saliency.shape[:-1])
        rows, cols = zip(*data[cluster_idxs])
        region_mask[rows, cols] = 1

        region_mask = cv2.blur(region_mask, (2, 2))
        thresh_image = cv2.bitwise_not(region_mask)
        thresh_image = thresh_image.astype(np.uint8)
        contour, hierarchy = cv2.findContours(thresh_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(thresh_image, [cnt], 0, 255, -1)
        region_mask = cv2.bitwise_not(thresh_image)

        ret, thresh = cv2.threshold(region_mask, 0, 255, cv2.THRESH_BINARY_INV)
        thresh = thresh.astype(np.uint8)

        mask = region_mask.copy()
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_regions = []
        for contour_index, contour in enumerate(contours):
            if hierarchy[0, contour_index, 3] != -1: # Remove regions that are completely inside another
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_x_units:
                rectangular_regions.append([slice(y, y+h), slice(x, x+w)])
                cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                eliminated_rectangular_regions.append([x, y, w, h])
        rectangular_regions_approximations.extend(rectangular_regions)

    if len(rectangular_regions_approximations) == 0:
        # Relax min_region_width constraint and Take one widest region from eliminated ones
        eliminated_rectangular_regions = sorted(eliminated_rectangular_regions, key=lambda el: el[2], reverse=True)
        if len(eliminated_rectangular_regions) == 0 and retry < 5:
            logger.debug("No Region extracted. Trying again with 5% lesser threshold. Retry count", retry)
            hp_copy = hp.copy()
            hp_copy["threshold_percentile"] -= 5
            hp_copy["min_cluster_size"] -= 5
            return get_rectangular_regions(example, saliency_copy, dataset_params, hp_copy, remove_silence=remove_silence,
                                           retry=retry + 1)
        elif len(eliminated_rectangular_regions) != 0:
            x, y, w, h = eliminated_rectangular_regions[0]
            rectangular_regions_approximations.append([slice(y, y + h), slice(x, x + w)])
        elif retry == 5:
            logger.debug("No Region extracted. Trying again with 5% lesser threshold. Retry count", retry)

    return rectangular_regions_approximations

