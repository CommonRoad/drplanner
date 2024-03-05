from abc import ABC, abstractmethod
import os


class DescriptionBase(ABC):
    domain = "base"

    def __init__(self):
        self.description = None

    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Abstract method to generate the description.
        Subclasses must implement this method.
        """
        pass

    def output_description(self, save_path="prompt/"):
        """
        Method to output the generated description to a text file.
        The file name can be specified; defaults to 'description.txt'.
        """
        filename = f"{self.domain}.txt"
        if self.description is not None:
            full_path = os.path.join(save_path, filename)
            with open(full_path, "w") as file:
                file.write(self.description)
            print(f"<output> Description saved to {full_path}.")
        else:
            print("<output> No description generated yet.")
