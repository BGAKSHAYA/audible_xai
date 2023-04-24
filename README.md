# Sonifying Explanations for Deep Neural Network Predictions on Audio Data

This repository contains code in the folder [AudibleXAI](AudibleXAI/) for the research paper on *Sonifying Explanations for Deep Neural Network Predictions on Audio Data*. 


### AudibleXAI directory structure

<ul>
<li>audio_utils - This folder contains files for audio amplification, two audio extraction strategies: audio identification and audio reconstruction</li>
<li>constants - Folder containing the constants files used throughout the application</li>
<li>dataset_preprocessing - It contains files to preprocess a dataset. We currently support AudioMNIST, GTZAN and Audio Set datasets. New datasets can be easily added either by using the provided pipeline or custom preprocessing pipeline </li>
<li>innvestigate_utils - Contains utility files for visualizing relevant features as heatmaps</li>
<li>model_training_utils - The model architectures we used for digit and gender classification tasks for AudioMNIST dataset </li>
<li>region_extraction_utils - Utility files for extracting relevant regions from the relevance scores obtained from attribution methods </li>
<li>services - Contains service files for loading dataset and model into the application, extracting relevances, relevant regions, audio, and performing the evaluation. </li>
<li>utils - Generic utility files for evaluation, visualizations </li> 
<li>main.py - Entry file for the project</li>
<li>path_configuration.json - Contains the paths to the datasets, models, and storing results</li>
<li>requirements.txt - File with the required packages and versions. Please install the python packages for the project through this file</li>
<li>run_configuration.json - JSON file containing the run options for the project.</li>
</ul>


### Run configuration

The following is a sample run_configuration.json file which is the basis for running our project with various settings. 

```javascript
{
  "path_configuration": "path_configuration.json", // path to the path_configuration.json file
  "dataset": "audio_mnist_digits", // Dataset to perform experiments
  "dataset_type": "spectrogram", // Dataset representation. Can either be spectrogram/ melspectrogram

  "run": { // Enable/ Disable each part of the pipeline accordingly 
    "preprocessing": false, //preprocess the dataset or not
    "saliency": true, // extract relevant features from attribution methods or not
    "region_extraction": true, // extract relevant regions from relevances or not
    "evaluation": true, // evaluate for the quantitative aspect of faithfulness or not
    "visualization": true 
  },
  "settings": { // Contains settings for each run option
    "miscellaneous": {
      "quick_analysis_mode": { // In the quick_analysis_mode, only load a subset of the dataset to run experiments
        "run" : true, // Enable this to get quicker results
        "max_per_class": 200 // maximum no. of files to load per class
      }
    },
    "preprocessing": { // If run option "preprocessing" is enabled, carry out the preprocessing for 
                       // the dataset and representations specified here.
      "audio_mnist_digits": ["spectrogram","melspectrogram"],
      "audioset" : ["melspectrogram"]
    },
    "saliency": {
      "methods": [ // The attributions/ saliency methods to run
        "gradient", "input_t_gradient", "INTEGRATED_GRADIENTS", "deconvnet",
        "deep_taylor", "lrp.sequential_preset_a",
        "GRAD_CAM", "OCCLUSION", "LIME"
      ],
      "save_saliency_heatmaps": false, 
      "reset": false // Reset the previous saliency results or not
    },
    "region_extraction": {
      "method": "RECTANGULAR_REGIONS", // Region extraction strategy.
      "search_space": { // The grid search space to perform hyperparameter tuning
        "threshold_percentile": [98, 95, 90, 80, 70],
        "min_cluster_size": [15, 20, 30, 40, 50, 60, 70], // HDBSCAN parameter
        "min_samples": [5, 10, 15, 20, 25, 30], // HDBSCAN parameter
        "max_inter_class_penalties": [0.5] // Penality term given to increase intra class similarity. 
                                           // This parameter is used when run_region_optimization is true
      },
      "region_tuning_mode": true, // Enable when tuning the hyperparameters
      "random_selection_comparison_runs": 5, // Number of random region selections to compare the relevant region selections
      "run_region_optimization": false, // If true, remove some region selections that hurt intra class similarity
      "save_audio": false, // Save audio of region selections
      "audio_extraction_strategy": "AUDIO_IDENTIFICATION", // Can either be AUDIO_INDENTIFICATION/ AUDIO_RECONSTRUCTION
      "reload_regions_last_run": false, // Reload the regions from the last run or extract regions from scratch
      "reset": false // Reset previous results for region selections or not
    },
    "evaluation": { 
      "method": "RECTANGULAR_REGIONS",
      "reset": false
    },
    "visualization": {
      "methods": [  
        "overview_saliencies", // Grid plot to visualize relevant features
        "overview_saliency_regions", // Grid plot to visualize relevant features and regions extracted
        "overview_mispredictions" // Grid plot to visualize features responsible for misprediction and regions extracted
      ]
    }
  }
}
```

### Project setup
$ pip install -r requirements.txt

### Terminal:
python3 main.py
