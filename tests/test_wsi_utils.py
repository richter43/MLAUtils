import json
import os
import sys
import unittest
import time

import matplotlib.pyplot as plt
from src import (DatasetManager, RenalCancerType, SlideMetadata,
                 get_label_from_path)

module_dir = os.path.dirname(__file__)
with open(os.path.join(module_dir, "paths.json")) as f:
    paths_json = json.load(f)
    XML_FILENAME = paths_json['XML_FILENAME']
    ccRCC_FILEPATH = paths_json["SLIDE_FILENAME"]

class TestWSIUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filepaths = [ccRCC_FILEPATH]
        cls.xml_paths = [XML_FILENAME]
        cls.tile_size = 224
        cls.slide_metadata_list = [SlideMetadata(wsi_path, xml_path=xml_path, label=get_label_from_path(wsi_path)) for wsi_path, xml_path in zip(cls.filepaths, cls.xml_paths)]

    def test_dataset_manager(self):
        dm = DatasetManager(self.slide_metadata_list, self.tile_size)

    def test_create_tensorflow_dataset(self):

        tic = time.perf_counter()

        dm = DatasetManager(self.slide_metadata_list, self.tile_size)
        dataset = dm.make_dataset()

        toc = time.perf_counter()
        print(f"Time taken to load dataset without removing white images: {toc - tic}")
        for image_batch, _ in dataset:
            plt.imshow(image_batch[0])
            plt.show()
            break

    def test_create_torch_dataset(self):
        dm = DatasetManager(self.slide_metadata_list, self.tile_size)
        dataset = dm.make_pytorch_dataset(remove_white=False)
        for image, label in dataset:
            plt.imshow(image.permute(1,2,0))
            plt.show()
            break

    def test_remove_white_torch(self):

        tic = time.perf_counter()

        dm = DatasetManager(self.slide_metadata_list, self.tile_size)
        dataset = dm.make_pytorch_dataset(remove_white=True)

        toc = time.perf_counter()
        print(f"Time taken to load dataset while removing white images: {toc - tic}")
        for image, label in dataset:
            plt.imshow(image.permute(1,2,0))
            plt.show()
            break