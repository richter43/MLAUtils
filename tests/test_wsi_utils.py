import sys
import os
import json
import unittest
from src import DatasetManager

module_dir = os.path.dirname(__file__)
with open(os.path.join(module_dir, "paths.json")) as f:
    paths_json = json.load(f)
    ccRCC_FILEPATH = paths_json["SLIDE_FILENAME"]

class TestWSIUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filepaths = [ccRCC_FILEPATH]
        cls.labels = ["ccRCC", "not-cancer"]
        cls.tile_size = 128

    def test_dataset_manager(self):
        dm = DatasetManager(self.filepaths, self.labels, self.tile_size)