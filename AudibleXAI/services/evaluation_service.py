import numpy as np
import pandas as pd
import os
from constants.evaluation_constants import AUDIO_TYPES
from constants.dataset_constants import DATASETS
from services.dataset_service import DatasetService
from services.file_service import FileService
from services.model_service import ModelService
from dataset_preprocessing.dataset_preprocessor import preprocess_example
from scipy import stats
import logging

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
logger = logging.getLogger("audible_xai")


class EvaluationService:
    def __init__(self, path_config, run_config):
        self.saliency_path = path_config["results"]["saliency_path"]
        self.regions_path = path_config["results"]["correct_regions_path"]
        self.evaluation_path = path_config["results"]["evaluation_path"]

        self.model = ModelService.instance.get_model()

        self.data = DatasetService.instance.get_dataset(filter_type="correct")
        self.dataset_filenames = DatasetService.instance.get_dataset_filenames(filter_type="correct")

        self.dataset_params = DatasetService.instance.get_dataset_params()
        self.dataset_type = DatasetService.instance.get_dataset_type()
        self.dataset_name = DatasetService.instance.get_dataset_name()
        self.class_name_to_index_mapper = DatasetService.instance.class_name_to_index_mapper()
        self.classnames = DatasetService.instance.get_classnames()

        self.file_service = FileService()
        self.class_names = list(self.data.keys())
        self.run_config = run_config

    def run(self, saliency_methods, random_selection_comparison_runs=5):
        evaluation_results = []
        for saliency_method in saliency_methods:
            region_extraction_results = np.load(os.path.join(self.regions_path, saliency_method + ".npy"),
                                                allow_pickle=True).item()

            logger.debug("Started evaluation for", saliency_method)
            result = self.evaluate_faithfulness(region_extraction_results, saliency_method,
                                                random_selection_comparison_runs)
            evaluation_results.append(result)
        evaluation_results = pd.DataFrame(
            evaluation_results, columns=["Saliency Method", "Examples with no regions",
                                         "Mean Actual Prediction score", "Mean Relevant Audio Prediction Score",
                                         "Mean Random Audio Prediction Score"
                                         ])
        self.file_service.create_directory(self.evaluation_path)
        runs_summary_path = os.path.join(self.evaluation_path, "runs_summary.csv")
        evaluation_results.to_csv(runs_summary_path, index=False)
        logger.debug(evaluation_results)

    def evaluate_faithfulness(self, region_extraction_results, saliency_method, random_selection_comparison_runs):
        no_examples_with_no_regions = 0
        prediction_scores = []

        for class_name in self.class_names:
            class_index = self.class_name_to_index_mapper[class_name]
            input_data, relevant_data, random_data = [], [], []
            for index, (example, filename) in enumerate(zip(self.data[class_name], self.dataset_filenames[class_name])):
                region_extraction_result = region_extraction_results[class_name][filename]
                if region_extraction_result is None:
                    no_examples_with_no_regions += 1
                    continue

                relevant_audio_path = region_extraction_result[AUDIO_TYPES.RELEVANT.value]
                random_audio_paths = region_extraction_result[AUDIO_TYPES.RANDOM.value]
                relevant_example = preprocess_example(relevant_audio_path, self.dataset_name,
                                                      self.dataset_type, self.dataset_params)
                if self.dataset_name != DATASETS.AUDIOSET.value:
                    relevant_example = relevant_example

                for random_run_index, random_audio_path in enumerate(random_audio_paths):
                    random_example = preprocess_example(random_audio_path, self.dataset_name,
                                                        self.dataset_type, self.dataset_params)
                    random_data.append(random_example)

                relevant_data.append(relevant_example)
                input_data.append(np.expand_dims(example, axis=0))

            input_data, relevant_data, random_data = np.array(input_data), np.array(relevant_data), np.array(random_data)
            if len(input_data) == 0:
                continue

            actual_prediction = self.model.predict(self.dataset_params.reshape_model_input(input_data), batch_size=64)
            relevant_prediction = self.model.predict(self.dataset_params.reshape_model_input(relevant_data),
                                                     batch_size=64)
            random_prediction = self.model.predict(self.dataset_params.reshape_model_input(random_data), batch_size=64)

            if self.dataset_name == DATASETS.AUDIOSET.value:
                actual_prediction = np.mean(actual_prediction.reshape((-1, 11, 521)), axis=1)
                relevant_prediction = np.mean(relevant_prediction.reshape((-1, 11, 521)), axis=1)
                random_prediction = np.mean(random_prediction.reshape((-1, 11, 521)), axis=1)

            actual_prediction_scores = np.repeat(actual_prediction[:, class_index], random_selection_comparison_runs)
            relevant_audio_prediction_scores = np.repeat(relevant_prediction[:, class_index],
                                                         random_selection_comparison_runs)
            random_audio_prediction_scores = random_prediction[:, class_index]

            prediction_scores.extend(list(zip(actual_prediction_scores, relevant_audio_prediction_scores,
                                              random_audio_prediction_scores)))
        prediction_scores = np.array(prediction_scores)
        stat, p = stats.ttest_rel(list(prediction_scores[:, 1]), list(prediction_scores[:, 2]))
        logger.debug('Paired t-test: stat=%.3f, p=%.3f' % (stat, p))
        logger.debug(prediction_scores[:, 1].mean(), prediction_scores[:, 2].mean())
        return [saliency_method, no_examples_with_no_regions,
                round(prediction_scores[:, 0].mean(), 2),
                round(prediction_scores[:, 1].mean(), 2), round(prediction_scores[:, 2].mean(), 2)]

