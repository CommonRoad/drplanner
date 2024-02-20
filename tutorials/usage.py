import pathlib

from commonroad.common.file_reader import CommonRoadFileReader

import drplanner
from drplanner.main import TemplateClass

scenario, planning_problem = CommonRoadFileReader(
    pathlib.Path(drplanner.__file__).parent.joinpath("./../scenarios/DEU_Guetersloh-15_2_T-1.xml")
).open()

crtemplate_object = TemplateClass(scenario)
