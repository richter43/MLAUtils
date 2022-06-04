import unittest
from src.annotation_utils_asap import *
import matplotlib.pyplot as plt
from typing import Optional
from openslide import OpenSlide


# Uncomment and define the file's information here
# SLIDE_FILENAME = 
# XML_FILENAME = 


class TestFrontier(unittest.TestCase):

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

    # def tearDown(self) -> None:
    #     self.frontier._purge_queue()
        
    # def tearDownClass() -> None:
    #     pass