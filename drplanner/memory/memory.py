import base64
import json
import os

import chromadb
import replicate
from chromadb.utils import embedding_functions
from utils.config import DrPlannerConfiguration


class FewShotMemory:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_storage = os.path.join(script_dir, "store")
        if not os.path.exists(path_to_storage):
            os.makedirs(path_to_storage, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path_to_storage)
        try:
            self.few_shot_collection = self.client.get_collection(name="few_shots")
        except ValueError as _:
            self.few_shot_collection = self.client.create_collection(name="few_shots")
            self.init_collection("few_shots")
            print("MEMORY: Initialized <few-shot> collection successfully!")
        try:
            self.diagnosis_collection = self.client.get_collection(name="diagnosis")
        except ValueError as _:
            self.diagnosis_collection = self.client.create_collection(name="diagnosis")
            self.init_collection("diagnosis")
            print("MEMORY: Initialized <diagnosis> collection successfully!")
        try:
            self.prescription_collection = self.client.get_collection(
                name="prescription"
            )
        except ValueError as _:
            self.prescription_collection = self.client.create_collection(
                name="prescription"
            )
            self.init_collection("prescription")
            print("MEMORY: Initialized <prescription> collection successfully!")
        try:
            self.plot_collection = self.client.get_collection(name="plot")
        except ValueError as _:
            self.plot_collection = self.client.create_collection(name="plot")
            path_to_plots = os.path.join(script_dir, "plots")
            filenames = [
                "DEU_Frankfurt-191_12_I-1.cr",
                "DEU_Frankfurt-11_8_I-1.cr",
                "DEU_Muc-19_1_I-1-1.cr",
                "DEU_Frankfurt-95_9_I-1.cr",
                "ESP_Mad-1_8_I-1-1.cr",
            ]

            for filename in filenames:
                fp = os.path.join(path_to_plots, filename + ".png")
                dc = os.path.join(path_to_plots, filename + ".txt")
                with open(dc, "r") as file:
                    cf_string = file.read()
                self.insert_image(fp, cf_string)
            print("MEMORY: Initialized <plot> collection successfully!")

    def insert_image(self, filepath, document: str):
        with open(filepath, "rb") as file:
            data = base64.b64encode(file.read()).decode("utf-8")
            input = f"data:application/octet-stream;base64,{data}"
        input = {"input": input}

        output = replicate.run(
            "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
            input=input,
        )
        print(f"generated embedding for {os.path.basename(filepath)}")
        collection = self.plot_collection
        docs = [document]
        embeddings = [output]
        ids = [f"plot{collection.count()}"]
        collection.add(ids=ids, embeddings=embeddings, documents=docs)

    def retrieve_with_image(self, filepath: str):
        with open(filepath, "rb") as file:
            data = base64.b64encode(file.read()).decode("utf-8")
            input = f"data:application/octet-stream;base64,{data}"
        input = {"input": input}

        output = replicate.run(
            "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
            input=input,
        )
        query = self.plot_collection.query(
            query_embeddings=[output], n_results=1, include=["documents"]
        )
        return query["documents"][0], query["ids"][0]

    def select_collection(self, collection_name):
        if collection_name == "few_shots":
            return self.few_shot_collection
        elif collection_name == "diagnosis":
            return self.diagnosis_collection
        elif collection_name == "prescription":
            return self.prescription_collection
        else:
            raise ValueError("This collection does not exist")

    def init_collection(self, collection_name: str):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        # read all predefined few-shots and add them to the collection
        path_to_few_shots = os.path.join(script_dir, "few_shots", collection_name)
        docs: list[str] = []
        embeddings = []
        ids: list[str] = []
        collection = self.select_collection(collection_name)

        with open(os.path.join(path_to_few_shots, "few_shots.json"), "r") as file:
            examples: list = json.load(file)["few_shots"]
            for data in examples:
                key: str = data["key"]
                value: str = data["value"]
                embedding = default_ef([key])[0]
                docs.append(value)
                embeddings.append(embedding)
                ids.append(f"{collection_name}{len(ids)}")

        collection.add(documents=docs, embeddings=embeddings, ids=ids)

    @staticmethod
    def preprocess_key(key) -> str:
        result = key
        if isinstance(key, list):
            result = ""
            for thing in key:
                if isinstance(thing, dict):
                    for a, b in thing.items():
                        result += f"{a}: {b}\n"
                else:
                    result += thing + "\n"
        return result

    def retrieve(self, key, collection_name="few_shots", n=1) -> list[str]:
        collection = self.select_collection(collection_name)
        key = self.preprocess_key(key)
        query = collection.query(query_texts=[key], n_results=n, include=["documents"])
        return query["documents"][0]

    def insert(self, key: str, value: str, collection_name="few_shots"):
        collection = self.select_collection(collection_name)
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        embeddings = [default_ef([key])[0]]
        ids = [f"{collection_name}{collection.count()}"]
        collection.add(documents=[value], embeddings=embeddings, ids=ids)
