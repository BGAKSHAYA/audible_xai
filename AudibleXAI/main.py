import os
import sys
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Set Tensorflow logging level to ERROR

logging.basicConfig(stream=sys.stdout,
                    filemode="a",
                    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.ERROR)

logger = logging.getLogger("audible_xai")
logger.setLevel(level=logging.DEBUG)

from dataset_preprocessing.dataset_preprocessor import DatasetPreprocessorService
from services.saliency_service import SaliencyService
from services.relevant_region_extractor_service import RelevantRegionsExtractorService
from services.mispredictions_region_extractor_service import MispredictionsRegionsExtractorService
from services.visualization_service import VisualizationService
from services.evaluation_service import EvaluationService
from services.dataset_service import DatasetService
from services.model_service import ModelService
import json

run_configuration = json.load(open("run_configuration.json"))
path_configuration = json.load(open(run_configuration["path_configuration"]))


if __name__ == "__main__":
    dataset = run_configuration["dataset"]
    dataset_type = run_configuration["dataset_type"]

    settings = run_configuration["settings"]
    runners = run_configuration["run"]

    miscellaneous_config = settings["miscellaneous"]
    preprocessing_config = settings["preprocessing"]
    saliency_config = settings["saliency"]
    region_extraction_config = settings["region_extraction"]
    evaluation_config = settings["evaluation"]
    visualization_config = settings["visualization"]

    quick_analysis_mode = miscellaneous_config["quick_analysis_mode"]

    if runners["preprocessing"]:
        dataset_preprocessor_service = DatasetPreprocessorService(path_configuration, preprocessing_config)
        dataset_preprocessor_service.run()

    path_config = path_configuration[dataset][dataset_type]

    model_weights_path = path_config["model"]["model_weights_path"]
    model_architecture_path = path_config["model"]["model_architecture_path"]

    saliency_methods = saliency_config["methods"]
    random_selection_comparison_runs = region_extraction_config["random_selection_comparison_runs"]

    ModelService.create_instance(model_weights_path, model_architecture_path)
    DatasetService.create_instance(path_config, dataset, dataset_type, quick_analysis_mode)

    if runners["saliency"]:
        saliency_service = SaliencyService(path_config, saliency_config)
        saliency_service.run(saliency_methods)

    if runners["region_extraction"]:
        relevant_regions_extractor_service = RelevantRegionsExtractorService(path_config, region_extraction_config)
        relevant_regions_extractor_service.run(saliency_methods)
        mispredictions_regions_extractor_service = MispredictionsRegionsExtractorService(path_config,
                                                                                         region_extraction_config)
        mispredictions_regions_extractor_service.run(saliency_methods)

    if runners["evaluation"]:
        evaluation_service = EvaluationService(path_config, evaluation_config)
        evaluation_service.run(saliency_methods, random_selection_comparison_runs)

    if runners["visualization"]:
        visualization_service = VisualizationService(path_config, visualization_config)
        visualization_service.run(saliency_methods)
