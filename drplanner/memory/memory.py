import base64
import os
from typing import Tuple

import chromadb
import replicate
from chromadb.utils import embedding_functions
from scipy.spatial.distance import cosine


class FewShotMemory:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_storage = os.path.join(script_dir, "store")
        self.separator = "@"  # todo: find a more elegant way
        self.threshold = 0.8

        if not os.path.exists(path_to_storage):
            os.makedirs(path_to_storage, exist_ok=True)

        self.client = chromadb.PersistentClient(path=path_to_storage)

        try:
            self.collection = self.client.get_collection(name="few_shots")
            print("[DrPlanner] MEMORY: Loaded existing collection successfully!")
        except ValueError as _:
            self.collection = self.client.create_collection(name="few_shots")
            print("[DrPlanner] MEMORY: Initialized missing collection successfully!")

    @staticmethod
    def embedd_text(text: str):
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        return default_ef([text])[0]

    @staticmethod
    def embedd_image(filepath):
        # from https://replicate.com/daanelson/imagebind
        with open(filepath, "rb") as file:
            data = base64.b64encode(file.read()).decode("utf-8")
            input = f"data:application/octet-stream;base64,{data}"
        input = {"input": input}

        try:
            output = replicate.run(
                "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
                input=input,
            )
            print(f"generated embedding for {os.path.basename(filepath)}")
        except Exception as _:
            raise ValueError(
                "DrPlanner failed to generate an image vector embedding. This is likely due to missing authentication data. To resolve this, include your REPLICATE_API_TOKEN as environment variable in the run configuration."
            )
        return output

    @staticmethod
    def similarity(a, b):
        """Compute cosine similarity of two vectors."""
        similarity = 1 - cosine(a, b)
        print(f"Similarity:{similarity}")
        return similarity

    def get_few_shots(
        self, path_to_plot: str, n: int, threshold=None
    ) -> list[Tuple[str, str]]:
        """Method to retrieve few-shot examples closely related to the input scenario plot."""
        results, _ = self.retrieve(path_to_plot, n=n, threshold=threshold)
        results = [x[0].split(self.separator) for x in results]
        return results[:n]

    def retrieve(self, image_file_path: str, n=1, threshold=None) -> Tuple[list, list]:
        """Returns the retrieved memories and the cached image embedding"""
        if n <= 0:
            return [], []
        if not threshold:
            threshold = self.threshold

        image_embedding = self.embedd_image(image_file_path)
        query = self.collection.query(
            query_embeddings=image_embedding,
            n_results=10,
            include=["documents", "metadatas", "embeddings"],
        )
        query_image_embeddings = query["embeddings"][0]
        query_ids = query["ids"][0]
        query_metadata = [x["total_cost"] for x in query["metadatas"][0]]
        query_documents = query["documents"][0]
        # combine all memory data into a single list
        query_data = list(
            zip(query_documents, query_image_embeddings, query_metadata, query_ids)
        )
        # filter out memories that do not meet the similarity threshold
        query_data = list(
            filter(
                lambda x: self.similarity(image_embedding, x[1]) >= threshold,
                query_data,
            )
        )
        # retrieve the remaining, most similar memories
        query_data.sort(key=lambda x: x[2], reverse=True)
        n = min(len(query_data), n)
        return query_data[:n], image_embedding

    def insert(
        self,
        diagnosis: str,
        cost_function: str,
        total_cost: float,
        image_file_path: str,
    ) -> bool:
        """
        Insert a new memory consisting of diagnosis and repair output.
        """
        # combine diagnosis and repair into a single text
        value = f"{diagnosis}{self.separator}{cost_function}"
        results, image_embedding = self.retrieve(image_file_path, threshold=0.95)
        if not image_embedding:
            image_embedding = self.embedd_image(image_file_path)

        # check whether a related memory already exists
        if results:
            result = results[0]
            result_total_cost = result[2]
            result_id = result[3]
            # if the old memory is inferior, replace it
            if total_cost > result_total_cost:
                self.collection.update(
                    documents=value,
                    embeddings=image_embedding,
                    metadatas={"total_cost": total_cost},
                    ids=result_id,
                )
                return True
            else:
                return False
        # else create a new memory
        else:
            idx = self.collection.count()
            result_id = f"nr{idx}"
            self.collection.add(
                documents=value,
                embeddings=image_embedding,
                metadatas={"total_cost": total_cost},
                ids=result_id,
            )
            return True

    def get_size(self):
        return self.collection.count()
