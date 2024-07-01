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
            self.collection = self.client.get_collection(name="few_shots")
        except ValueError as _:
            self.collection = self.client.create_collection(name="few_shots")
            self.init_collection()

    def init_collection(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_ef = embedding_functions.DefaultEmbeddingFunction()

        # read all predefined few-shots and add them to the collection
        path_to_few_shots = os.path.join(script_dir, "static_few_shots")
        few_shot_docs: list[str] = []
        few_shot_embeddings = []
        few_shot_ids: list[str] = []
        counter = 0
        for file_name in os.listdir(path_to_few_shots):
            with open(os.path.join(path_to_few_shots, file_name), "r") as file:
                examples: list = json.load(file)["few_shots"]
                for data in examples:
                    key: str = data["key"]
                    value: str = data["value"]
                    embedding = default_ef([key])[0]
                    few_shot_docs.append(value)
                    few_shot_embeddings.append(embedding)
                    few_shot_ids.append(f"id{counter}")
                    counter += 1

        self.collection.add(documents=few_shot_docs, embeddings=few_shot_embeddings, ids=few_shot_ids)

    def retrieve(self, diagnosis: str, prescription: str) -> list[str]:
        key = f"{diagnosis}: {prescription}"
        result = self.collection.query(
            query_texts=[key],
            n_results=3,
            include=["documents"]
        )
        return result["documents"][0]
