import os
import sys
sys.path.insert(0, os.path.join('/scratch/python_envs/annalyzer/python/lib/python3.8/site-packages/'))
sys.path.insert(1, os.path.join('/scratch/bindu/python/lib/python3.8/site-packages/'))

from pydub import AudioSegment
import math
import glob
import logging
logger = logging.getLogger('audible_xai')


class AudioSplitter:
    """
    Helper class to split the audio files of a dataset with the duration of each split as "seconds_per_split"
    """
    def __init__(self, source_folder, destination_folder):
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def run(self, audio_duration=30, seconds_per_split=3):
        folders = []
        for folder in os.listdir(self.source_folder):
            # only process folders
            if not os.path.isdir(os.path.join(self.source_folder, folder)):
                continue
            folders.append(folder)

        src_dst_list = [(os.path.join(self.source_folder, folder), os.path.join(self.destination_folder, folder))
                        for folder in sorted(folders)]

        for src_dst in src_dst_list:
            src, dst = src_dst
            logger.info("Audio Splitting started for {}".format(src))
            if not os.path.exists(dst):
                os.makedirs(dst)

            for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
                audio = AudioSegment.from_wav(filepath)
                total_seconds = math.floor(audio.duration_seconds)
                if total_seconds != audio_duration:
                    logger.error("Audio duration should be {}s but got {}s".format(audio_duration, total_seconds))
                    raise Exception("Audio duration should be {}s but got {}s".format(audio_duration, total_seconds))

                filename = os.path.splitext(os.path.basename(filepath))[0]
                for index, time in enumerate(range(0, total_seconds, seconds_per_split)):
                    split_filename = filename + '_' + str(index) + ".wav"
                    start_time = time * 1000
                    end_time = (time + seconds_per_split) * 1000
                    split_audio = audio[start_time: end_time]
                    split_audio.export(os.path.join(dst, split_filename), format="wav")

