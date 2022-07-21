import random
import numpy as np
import math


def get_slice_area(row_slice, col_slice):
    area = (row_slice.stop - row_slice.start) * (col_slice.stop - col_slice.start)  # length*breadth
    return area


def get_overlapping_relevant_regions(random_regions, relevant_region_row_slice, relevant_region_col_slice):
    indices = []
    for index, (random_region_row_slice, random_region_col_slice) in enumerate(random_regions):
        overlapping_row = relevant_region_row_slice.start >= random_region_row_slice.start and \
                          relevant_region_row_slice.stop <= random_region_row_slice.stop
        overlapping_col = relevant_region_col_slice.start >= random_region_col_slice.start and \
                          relevant_region_col_slice.stop <= random_region_col_slice.stop
        if overlapping_row and overlapping_col:
            indices.append(index)
    return indices


def split_space(space, row_slice, col_slice):  # Split the space that doesn't contain the region
    row_wise = row_slice.stop - row_slice.start > col_slice.stop - col_slice.start
    if row_wise:
        result = [(space[0], slice(space[1].start, col_slice.start)),
                  (space[0], slice(col_slice.stop, space[1].stop)),
                  (slice(space[0].start, row_slice.start), col_slice),
                  (slice(row_slice.stop, space[0].stop), col_slice)]

    else:
        result = [(slice(space[0].start, row_slice.start), space[1]),
                  (slice(row_slice.stop, space[0].stop), space[1]),
                  (row_slice, slice(space[1].start, col_slice.start)),
                  (row_slice, slice(col_slice.stop, space[1].stop))]
    return result


def get_smallest_possible_slice(slices, row_len, col_len):
    min_slice_index, min_area = None, math.inf
    possible_slices = []
    for index, (row_slice, col_slice) in enumerate(slices):
        slice_row_len = row_slice.stop - row_slice.start
        slice_col_len = col_slice.stop - col_slice.start

        if slice_row_len < row_len or slice_col_len < col_len:  # Slice is too small to place the random region
            continue

        possible_slices.append(index)
        slice_area = get_slice_area(row_slice, col_slice)
        if slice_area < min_area:
            min_slice_index, min_area = index, slice_area
    # Uncomment to return any possible slice
    # return random.choice(possible_slices) if len(possible_slices) > 0 else None
    return min_slice_index


def get_random_region(row_choices, col_choices, row_len, col_len):
    random_row_start = random.choice(row_choices)
    random_row_end = random_row_start + row_len
    random_row_slice = slice(random_row_start, random_row_end)

    random_col_start = random.choice(col_choices)
    random_col_end = random_col_start + col_len
    random_col_slice = slice(random_col_start, random_col_end)
    random_region = [random_row_slice, random_col_slice]
    return random_region


def pseudo_random_regions_generator(relevant_regions, dataset_shape):
    """
    The method generates pseudo random regions that does not overlap with any relevant region
    We first split the total input space into all possible slices that does not relevant regions and then randomly place
    random regions in those possible slices. If there are many relevant regions, then its often not possible to 
    generate same no. of random regions as relevant regions
    """
    random_regions = []
    possible_slices = [[slice(0, dataset_shape[0]), slice(0, dataset_shape[1])]]

    for row_slice, col_slice in relevant_regions:
        indices = get_overlapping_relevant_regions(possible_slices, row_slice, col_slice)
        resultant_slices = []
        for index in indices:
            slices = split_space(possible_slices[index], row_slice, col_slice)
            resultant_slices.extend(slices)
        possible_slices = [value for index, value in enumerate(possible_slices) if index not in indices]
        possible_slices.extend(resultant_slices)

    for index, (row_slice, col_slice) in enumerate(relevant_regions):
        row_len = row_slice.stop - row_slice.start
        col_len = col_slice.stop - col_slice.start
        region_index = get_smallest_possible_slice(possible_slices, row_len, col_len)

        if region_index is None:  # Not possible to place a random region case
            continue

        possible_slice = possible_slices[region_index]
        possible_slice_row_slice, possible_slice_col_slice = possible_slice

        # Now place it somewhere in the possible slice
        row_choices = list(range(possible_slice_row_slice.start, possible_slice_row_slice.stop - row_len + 1))
        col_choices = list(range(possible_slice_col_slice.start, possible_slice_col_slice.stop - col_len + 1))
        random_region = get_random_region(row_choices, col_choices, row_len, col_len)
        random_regions.append(random_region)

        resultant_slices = split_space(possible_slices[region_index], random_region[0], random_region[1])
        del possible_slices[region_index]
        possible_slices.extend(resultant_slices)
    return random_regions


def true_random_regions_generator(relevant_regions, dataset_shape):
    """
    The method generates true random regions irrespective of whether it overlaps with any relevant region i.e unlike
    pseudo_random_regions_generator method
    """
    random_regions = []
    for index, (row_slice, col_slice) in enumerate(relevant_regions):
        row_len = row_slice.stop - row_slice.start
        col_len = col_slice.stop - col_slice.start
        row_choices = list(range(0, dataset_shape[0] - row_len + 1))
        col_choices = list(range(0, dataset_shape[1] - col_len + 1))
        random_region = get_random_region(row_choices, col_choices, row_len, col_len)
        random_regions.append(random_region)
    return random_regions


def mae(a, b):  # Mean Absolute Error
    result = (np.abs(a - b)).mean()
    return result


def mse(a, b):  # Mean Squared Error
    result = (np.square(a - b)).mean()
    return result


def se(a, b):  # Sum of squared errors
    result = np.sum(np.square(a - b))
    return result


