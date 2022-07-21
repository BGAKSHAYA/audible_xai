import numpy as np
import tensorflow as tf
import soundfile as sf
import pandas as pd
import csv
import os
import librosa
from dataclasses import dataclass
from services.file_service import FileService

if int(tf.__version__.split('.')[0]) == 2:
    tf_version = 2
else:
    tf_version = 1
    session = tf.Session()

# Code adapted from https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
# Copyright 2019 The TensorFlow Authors All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


def pad_waveform(waveform, params):
    """ Pads waveform with silence if needed to get an integral number of patches."""
    # In order to produce one patch of log mel spectrogram input to YAMNet, we
    # need at least one patch window length of waveform plus enough extra samples
    # to complete the final STFT analysis window.
    min_waveform_seconds = (
            params.patch_window_seconds +
            params.stft_window_seconds - params.stft_hop_seconds)
    min_num_samples = tf.cast(min_waveform_seconds * params.sr, tf.int32)
    num_samples = tf.shape(waveform)[0]
    num_padding_samples = tf.maximum(0, min_num_samples - num_samples)

    # In addition, there might be enough waveform for one or more additional
    # patches formed by hopping forward. If there are more samples than one patch,
    # round up to an integral number of hops.
    num_samples = tf.maximum(num_samples, min_num_samples)
    num_samples_after_first_patch = num_samples - min_num_samples
    hop_samples = tf.cast(params.patch_hop_seconds * params.sr, tf.int32)
    num_hops_after_first_patch = tf.cast(tf.math.ceil(
        tf.cast(num_samples_after_first_patch, tf.float32) /
        tf.cast(hop_samples, tf.float32)), tf.int32)
    num_padding_samples += (
            hop_samples * num_hops_after_first_patch - num_samples_after_first_patch)

    padded_waveform = tf.pad(waveform, [[0, num_padding_samples]],
                             mode='CONSTANT', constant_values=0.0)
    padded_waveform = padded_waveform.eval(session=session)
    return padded_waveform


def get_magnitude_spectrogram(waveform, params):
    # waveform = pad_waveform(waveform, params)
    # waveform has shape [<# samples>]
    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    magnitude_spectrogram = tf.abs(tf.signal.stft(
        signals=waveform,
        frame_length=params.win_length,
        frame_step=params.hop_length,
        fft_length=params.n_fft))
    # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]
    return magnitude_spectrogram


def magnitude_spectrogram_to_log_mel_spectrogram_patches(magnitude_spectrogram, params):
    """ Compute log mel spectrogram patches of a 1-D waveform."""
    # Convert spectrogram into log mel spectrogram.
    num_spectrogram_bins = tf.shape(magnitude_spectrogram)[1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.mel_bands,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.sr,
        lower_edge_hertz=params.mel_min_hz,
        upper_edge_hertz=params.mel_max_hz)

    mel_spectrogram = tf.matmul(
        magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.log_offset)
    # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]

    # Frame spectrogram (shape [<# STFT frames>, params.mel_bands]) into patches
    # (the input examples). Only complete frames are emitted, so if there is
    # less than params.patch_window_seconds of waveform then nothing is emitted
    # (to avoid this, zero-pad before processing).
    spectrogram_hop_length_samples = int(
        round(params.sr * params.stft_hop_seconds))
    spectrogram_sample_rate = params.sr / spectrogram_hop_length_samples
    patch_window_length_samples = int(
        round(spectrogram_sample_rate * params.patch_window_seconds))
    patch_hop_length_samples = int(
        round(spectrogram_sample_rate * params.patch_hop_seconds))

    # For simplicity in generating audio explanations, hop_length = frame_length
    features = tf.signal.frame(
        signal=log_mel_spectrogram,
        frame_length=patch_window_length_samples,
        frame_step=patch_window_length_samples,  # patch_hop_length_samples,
        pad_end=True,
        pad_value=0,
        axis=0)

    # features has shape [<# patches>, <# STFT frames in an patch>, params.mel_bands]
    return log_mel_spectrogram, features


def preprocess_audioset_example(waveform, params):
    waveform = waveform.astype(np.float32)
    spectrogram = get_magnitude_spectrogram(waveform, params)
    log_mel_spectrogram, features = magnitude_spectrogram_to_log_mel_spectrogram_patches(spectrogram, params)
    if tf_version == 1:
        features = features.eval(session=session)
    else:
        features = features.numpy()
    return features


def restructure_audio_files():
    """
     Helper method to restructure audio files for AudioSet
     AudioSet Dataset we used is obtained from https://www.kaggle.com/datasets/zfturbo/audioset-valid
     Once the dataset is extracted, we restructure the files, obtain class labels and place in the data folder
    """
    root = "E:/Master Thesis/Code/data/datasets/audioset"
    source_folder = "valid_wav"
    destination_folder = "data_more"
    meta_file = "valid.csv"
    classname_mapper_file = "classname_mapper.csv"
    file_service = FileService()

    meta = pd.read_csv(os.path.join(root, meta_file))
    meta["positive_labels"] = list(map(lambda labels: labels.split(",")[0], meta["positive_labels"]))
    meta = meta.groupby('positive_labels')

    classname_df = pd.read_csv(os.path.join(root, classname_mapper_file))
    # classname_mapper = dict(zip(classname_mapper["mid"], classname_mapper["display_name"]))

    remove_groups = [group for group in meta.groups if group not in list(classname_df["mid"])]
    for remove_group in remove_groups:
        del meta.groups[remove_group]

    # classnames = classname_df["display_name"]
    classnames = ["Alarm", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Gunshot, gunfire", "Whistling",
                  "Smoke detector, smoke alarm", "Heart sounds, heartbeat", "Bark",
                  "Guitar", "Drum", "Organ", "Harp", "Brass", "Blues", "Country"]

    positions = classname_df["display_name"].isin(classnames)
    groups = classname_df["mid"][positions]
    classnames = classname_df["display_name"][positions]

    file_service.create_directory(root, destination_folder, classnames)
    source_path = os.path.join(root, source_folder)

    for classname, group in zip(classnames, groups):
        source_file_names = meta.get_group(group)["YTID"].to_list()[:30]  # Get youtube filenames of a class
        source_file_names = [filename + ".wav" for filename in source_file_names]
        destination_file_names = [classname + "." + str(index) + ".wav" for index in range(len(source_file_names))]

        destination_path = os.path.join(root, destination_folder, classname)
        file_service.copy_files(source_path, destination_path, source_file_names, destination_file_names)


# restructure_audio_files()
