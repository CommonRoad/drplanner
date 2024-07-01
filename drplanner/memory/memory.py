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

    def retrieve(self, summary: list[dict[str, str]]) -> set[str]:
        result: set[str] = set()
        docs: list[list[str]] = []

        # retrieve all relevant documents
        for data in summary:
            diagnosis = data["diagnosis"]
            prescription = data["prescription"]
            key = f"{diagnosis}: {prescription}"
            query = self.collection.query(
                query_texts=[key],
                n_results=3,
                include=["documents"]
            )
            docs.append(query["documents"][0])

        assert len(docs) == len(summary)
        # pick three
        index = 0
        while len(result) < 3:
            # check if there are no more docs
            if sum([len(x) for x in docs]) <= 0:
                break

            documents = docs[index]
            if len(documents) > 0:
                result.add(documents.pop(0))
            index = (index + 1) % len(docs)
        return result
