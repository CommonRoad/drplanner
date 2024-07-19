import json
import os

import chromadb
from chromadb.utils import embedding_functions


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
        try:
            self.diagnosis_collection = self.client.get_collection(name="diagnosis")
        except ValueError as _:
            self.diagnosis_collection = self.client.create_collection(name="diagnosis")
            self.init_collection("diagnosis")
        try:
            self.prescription_collection = self.client.get_collection(
                name="prescription"
            )
        except ValueError as _:
            self.prescription_collection = self.client.create_collection(
                name="prescription"
            )
            self.init_collection("prescription")

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
        for file_name in os.listdir(path_to_few_shots):
            with open(os.path.join(path_to_few_shots, file_name), "r") as file:
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

    def insert(self, key, value, collection_name="few_shots"):
        collection = self.select_collection(collection_name)
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        embeddings = [default_ef([key])[0]]
        ids = [f"{collection_name}{collection.count()}"]
        collection.add(documents=[value], embeddings=embeddings, ids=ids)
