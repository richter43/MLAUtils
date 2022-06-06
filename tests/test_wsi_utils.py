import sys
import os
import json
import unittest
from src import DatasetManager, SlideMetadata, RenalCancerType, get_label_from_path
import matplotlib.pyplot as plt

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
        cls.tile_size = 128
        cls.slide_metadata_list = [SlideMetadata(wsi_path, xml_path=xml_path, label=get_label_from_path(wsi_path)) for wsi_path, xml_path in zip(cls.filepaths, cls.xml_paths)]

    def test_dataset_manager(self):
        dm = DatasetManager(self.slide_metadata_list, self.tile_size)

    def test_create_dataset(self):
        dm = DatasetManager(self.slide_metadata_list, self.tile_size)
        dataset = dm.make_dataset()
        for image_batch, label_batch in dataset:
            plt.imshow(image_batch[0])
            plt.show()
            break