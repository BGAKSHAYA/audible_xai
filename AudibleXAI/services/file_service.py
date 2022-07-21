import logging
import os
import json
import configparser
import shutil


logger = logging.getLogger('audible_xai')


class FileService(object):
    @staticmethod
    def create_directory(path, directory_name=None, class_names=None):
        logger.info("create_directory service is started")
        if not os.path.isdir(path):
            os.makedirs(path)

        if directory_name is not None:
            directory_path = os.path.join(path, directory_name)
            if not os.path.isdir(directory_path):
                os.makedirs(directory_path)

        if class_names is not None:
            for class_name in class_names:
                class_path = os.path.join(path, directory_name, str(class_name))
                if not os.path.isdir(class_path):
                    os.makedirs(class_path)

    @staticmethod
    def remove_directory_contents(path, directory_name):
        directory_path = os.path.join(path, str(directory_name))
        for root, dirs, files in os.walk(directory_path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    @staticmethod
    def copy_files(source_path, destination_path, source_file_names, destination_file_names):
        for source_file_name, destination_file_name in zip(source_file_names, destination_file_names):
            source_file_path = os.path.join(source_path, source_file_name)
            destination_file_path = os.path.join(destination_path, destination_file_name)

            if not FileService.isfile(source_file_path):
                continue
            shutil.copyfile(source_file_path, destination_file_path)

    @staticmethod
    def isfile(file_path):
        isfile = os.path.isfile(file_path)
        return isfile

    @staticmethod
    def basename(file_path):
        basename_without_extension = os.path.basename(file_path.replace("\\", "/")).split(".")[0]
        return basename_without_extension
