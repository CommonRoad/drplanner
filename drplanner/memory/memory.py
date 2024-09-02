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
        self.separator = '@'
        self.threshold = 0.9
        if not os.path.exists(path_to_storage):
            os.makedirs(path_to_storage, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path_to_storage)
        # try:
        #     self.emergency_collection = self.client.get_collection(name="emergency_prescriptions")
        # except ValueError as _:
        #     self.emergency_collection = self.client.create_collection(name="emergency_prescriptions")
        #     self.init_collection("emergency_prescriptions")
        #     print("MEMORY: Initialized <few-shot> collection successfully!")
        try:
            self.collection = self.client.get_collection(name="few_shots")
        except ValueError as _:
            self.collection = self.client.create_collection(name="few_shots")
            path_to_plots = os.path.join(script_dir, "plots")
            filenames = [
                "DEU_Frankfurt-191_12_I-1.cr",
                "DEU_Frankfurt-11_8_I-1.cr",
                "DEU_Muc-19_1_I-1-1.cr",
                "DEU_Lohmar-34_1_I-1-1.cr",
                "DEU_Frankfurt-95_9_I-1.cr",
                "ESP_Mad-1_8_I-1-1.cr",
            ]
            for filename in filenames:
                fp = os.path.join(path_to_plots, filename + ".png")
                dc = os.path.join(path_to_plots, filename + ".txt")
                with open(dc, "r") as file:
                    cf_string = file.read()
                inserted = self.insert(filename, cf_string, 100, cf_string, fp)
                print(f"{filename}:{inserted}")
            print("MEMORY: Initialized collection successfully!")
        # try:
        #     self.prescription_collection = self.client.get_collection(
        #         name="prescription"
        #     )
        # except ValueError as _:
        #     self.prescription_collection = self.client.create_collection(
        #         name="prescription"
        #     )
        #     # self.init_collection("prescription")
        #     print("MEMORY: Initialized <prescription> collection successfully!")
        # try:
        #     self.plot_collection = self.client.get_collection(name="plot")
        # except ValueError as _:
        #     self.plot_collection = self.client.create_collection(name="plot")
        #     path_to_plots = os.path.join(script_dir, "plots")
        #     filenames = [
        #         "DEU_Frankfurt-191_12_I-1.cr",
        #         "DEU_Frankfurt-11_8_I-1.cr",
        #         "DEU_Muc-19_1_I-1-1.cr",
        #         "DEU_Frankfurt-95_9_I-1.cr",
        #         "ESP_Mad-1_8_I-1-1.cr",
        #     ]
#
        #     for filename in filenames:
        #         fp = os.path.join(path_to_plots, filename + ".png")
        #         dc = os.path.join(path_to_plots, filename + ".txt")
        #         with open(dc, "r") as file:
        #             cf_string = file.read()
        #         self.insert_image(fp, cf_string)
        #     print("MEMORY: Initialized <plot> collection successfully!")

    # def insert_image(self, filepath, document: str):
    #     collection = self.plot_collection
    #     docs = [document]
    #     embeddings = [embedd_image(filepath)]
    #     ids = [f"plot{collection.count()}"]
    #     collection.add(ids=ids, embeddings=embeddings, documents=docs)
#
    # def retrieve_with_image(self, filepath: str):
    #     query = self.plot_collection.query(
    #         query_embeddings=[embedd_image(filepath)], n_results=1, include=["documents"]
    #     )
    #     return query["documents"][0], query["ids"][0]

    # def select_collection(self, collection_name):
    #     if collection_name == "diagnosis":
    #         return self.diagnosis_collection
    #     elif collection_name == "prescription":
    #         return self.prescription_collection
    #     else:
    #         raise ValueError("This collection does not exist")

    # def init_collection(self, collection_name: str):
    #     script_dir = os.path.dirname(os.path.abspath(__file__))
    #     default_ef = embedding_functions.DefaultEmbeddingFunction()
    #     # read all predefined few-shots and add them to the collection
    #     path_to_few_shots = os.path.join(script_dir, "few_shots", collection_name)
    #     docs: list[str] = []
    #     embeddings = []
    #     ids: list[str] = []
    #     collection = self.select_collection(collection_name)
#
    #     with open(os.path.join(path_to_few_shots, "few_shots.json"), "r") as file:
    #         examples: list = json.load(file)["few_shots"]
    #         for data in examples:
    #             key: str = data["key"]
    #             value: str = data["value"]
#
    #             embedding = default_ef([key])[0]
    #             docs.append(value)
    #             embeddings.append(embedding)
    #             ids.append(f"{collection_name}{len(ids)}")
    #     collection.add(documents=docs, embeddings=embeddings, ids=ids)

    # @staticmethod
    # def preprocess_key(key) -> str:
    #     result = key
    #     if isinstance(key, list):
    #         result = ""
    #         for thing in key:
    #             if isinstance(thing, dict):
    #                 for a, b in thing.items():
    #                     result += f"{a}: {b}\n"
    #             else:
    #                 result += thing + "\n"
    #     return result

    @staticmethod
    def embedd_text(text: str):
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        return default_ef([text])[0]

    @staticmethod
    def embedd_image(filepath):
        with open(filepath, "rb") as file:
            data = base64.b64encode(file.read()).decode("utf-8")
            input = f"data:application/octet-stream;base64,{data}"
        input = {"input": input}

        output = replicate.run(
            "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
            input=input,
        )
        print(f"generated embedding for {os.path.basename(filepath)}")
        return output

    @staticmethod
    def combine(embedding1: list, embedding2: list) -> list:
        embedding1.extend(embedding2)
        return embedding1

    @staticmethod
    def similarity(a, b):
        similarity = 1 - cosine(a, b)
        print(f"Similarity:{similarity}")
        return similarity

    def get_few_shots(self, evaluation: str, path_to_plot: str, n: int) -> list[Tuple[str, str]]:
        results, _ = self.retrieve(evaluation, path_to_plot, n=n)
        results = [x[0].split(self.separator) for x in results]
        return results[:n]

    def retrieve(self, text: str, image_file_path: str, n=1) -> Tuple[list, list]:
        if n <= 0:
            return [], []

        text_embedding = self.embedd_text(text)
        image_embedding = self.embedd_image(image_file_path)
        query = self.collection.query(query_embeddings=image_embedding, n_results=max(10, n), include=["documents", "metadatas", "embeddings"])
        query_image_embeddings = query["embeddings"][0]
        query_metadata = query["metadatas"][0]
        query_text_embeddings = [self.embedd_text(x["text"]) for x in query_metadata]
        query_documents = query["documents"][0]
        query_data = list(zip(query_documents, query_text_embeddings, query_image_embeddings, query_metadata))
        query_data = list(filter(lambda x: self.similarity(image_embedding, x[2]) >= self.threshold, query_data))
        query_data.sort(key=lambda x: self.similarity(text_embedding, x[1]))
        n = min(len(query_data), n)
        print(f"N:{n}")
        return query_data[:n], image_embedding

    def insert(self, diagnosis: str, cost_function: str, total_cost: float, text: str, image_file_path: str) -> bool:
        value = f"{diagnosis}{self.separator}{cost_function}"
        doc_id = f"few_shot{self.collection.count()}"
        results, image_embedding = self.retrieve(text, image_file_path)
        if results:
            result = results[0]
            result_total_cost = result[3]["total_cost"]
        else:
            result_total_cost = total_cost + 1

        if not image_embedding:
            image_embedding = self.embedd_image(image_file_path)

        if total_cost < result_total_cost:
            self.collection.add(documents=value, embeddings=image_embedding, metadatas={"text": text, "total_cost": total_cost}, ids=doc_id)
            return True
        else:
            return False
