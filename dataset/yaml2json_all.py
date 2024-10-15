import yaml
import json
import os


def yaml_to_json(yaml_file, json_file):
    # Read the YAML file
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    # Convert the YAML content to JSON and write to a file
    with open(json_file, "w") as file:
        json.dump(yaml_content, file, indent=4)


def convert_all_yaml_in_folder(folder_path, yaml_folder="raw", json_folder="jsons"):
    # Loop through each file in the directory
    yaml_folder = os.path.join(folder_path, yaml_folder)
    for filename in os.listdir(yaml_folder):
        if filename.endswith(".yaml"):
            # Construct the full file paths
            yaml_file = os.path.join(folder_path, yaml_folder, filename)
            json_file = os.path.join(
                folder_path, json_folder, filename.replace(".yaml", ".json")
            )
            print(yaml_file)
            # Convert the YAML file to JSON format
            yaml_to_json(yaml_file, json_file)


# Replace 'path_to_folder' with the path of your folder containing the YAML files
convert_all_yaml_in_folder("./")
