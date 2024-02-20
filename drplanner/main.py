from commonroad.scenario.scenario import Scenario


class TemplateClass:
    """
    This is a class in the CommonRoad Template repository.
    """

    def __init__(self, scenario: Scenario):
        """
        Initialize TemplateClass object

        :param scenario: CommonRoad Scenario
        """
        self.scenario = scenario

    def return_number_of_lanelets(self) -> int:
        """
        Compute number of lanelets of scenario contained in TemplateClass
        object.

        :return: number of lanelets
        """
        number_of_lanelets = len(self.scenario.lanelet_network.lanelets)
        return number_of_lanelets
