from region_extraction_utils.timeframe import get_timeframe
from region_extraction_utils.rectangle_regions import get_rectangular_regions
from enum import Enum


REGION_EXTRACTION_METHODS = dict(
    TIMEFRAME=get_timeframe,
    RECTANGULAR_REGIONS=get_rectangular_regions,
)

REGION_AUDIO_AMPLIFICATION_FACTOR = 10

MIN_TIME_DURATION_REGION = 100  # Min time duration for a region is 100ms


class AUDIO_EXTRACTION_STRATEGY(Enum):
    AUDIO_IDENTIFICATION = "AUDIO_IDENTIFICATION"
    AUDIO_RECONSTRUCTION = "AUDIO_RECONSTRUCTION"
