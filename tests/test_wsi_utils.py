import sys
import os
import json
import unittest
from src import DatasetManager, SlideMetadata

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
        cls.tile_size = 1024

    def test_dataset_manager(self):

        slide_metadata_list = [SlideMetadata(wsi_path, xml_path=xml_path) for wsi_path, xml_path in zip(self.filepaths, self.xml_paths)]
        dm = DatasetManager(slide_metadata_list, self.tile_size)