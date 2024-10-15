import yaml
import json


def yaml_to_json(yaml_file, json_file):
    # Read the YAML file
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    # Convert the YAML content to JSON and write to a file
    with open(json_file, "w") as file:
        json.dump(yaml_content, file, indent=4)


id = 9192001
# Replace 'path/to/yaml.yaml' and 'path/to/json.json' with your file paths
yaml_to_json(f"./raw/{str(id)}.yaml", f"./jsons/{str(id)}.json")
