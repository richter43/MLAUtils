import unittest
import os
from src.annotation_utils_asap import *
import matplotlib.pyplot as plt
from typing import Optional
from openslide import OpenSlide
import json

module_dir = os.path.dirname(__file__)
with open(os.path.join(module_dir, "paths.json")) as f:
    paths_json = json.load(f)
    XML_FILENAME = paths_json['XML_FILENAME']
    SLIDE_FILENAME = paths_json['SLIDE_FILENAME']

# No assertions are established given that a standard testing image should be agreed upon

class TestAnnotationAsap(unittest.TestCase):

    slide: Optional[OpenSlide] = None

    def setUp(self) -> None:
        if self.slide is None:
            self.slide = OpenSlide(SLIDE_FILENAME)
        
    def test_get_points_xml(self) -> None:
        get_points_xml_asap(XML_FILENAME)

    def test_get_points_base_asap(self) -> None:
        get_points_base_asap(XML_FILENAME)

    def test_annotation_mask_asap(self) -> None:
        downsample = 0.1
        get_annotation_mask_asap(XML_FILENAME, self.slide, downsample)

    def test_overlap_asap(self) -> None:
        downsample = 0.1
        mask = get_annotation_mask_asap(XML_FILENAME, self.slide, downsample)
        img = overlap_asap(self.slide, mask, downsample)
        plt.imshow(img)
        plt.show()