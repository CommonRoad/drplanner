import pathlib
import unittest

from commonroad.common.file_reader import CommonRoadFileReader

import drplanner
from drplanner.main import TemplateClass


class TemplateClassTest(unittest.TestCase):
    def setUp(self):
        scenario, _ = CommonRoadFileReader(
            pathlib.Path(drplanner.__file__).parent.joinpath("./../scenarios/DEU_Guetersloh-15_2_T-1.xml")
        ).open()
        self.object = TemplateClass(scenario)

    def test_number_of_lanelets(self):
        self.assertEqual(self.object.return_number_of_lanelets(), 106)
