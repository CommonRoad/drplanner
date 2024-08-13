import os
import textwrap
from drplanner.utils.config import DrPlannerConfiguration
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document


class PlanningMemory:

    def __init__(self, db_path=None) -> None:
        config = DrPlannerConfiguration()
        db_path = os.path.join("./db", "chroma_24_mem/") if db_path is None else db_path
        self.embedding = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding, persist_directory=db_path
        )

    def retrieveMemory(self, prompt_planner: str, top_k):
        # search for similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(
            prompt_planner, top_k
        )
        fewshot_results = []
        top_three_similar_memories = self.top_three_similar_memories(similarity_results)
        for idx in range(0, len(top_three_similar_memories)):
            # print("The score for memory:",str(top_three_similar_memories[idx][1]))
            fewshot_results.append(top_three_similar_memories[idx][0].metadata)
        return fewshot_results

    def top_three_similar_memories(self, memories: any):
        sorted_memories = sorted(memories, key=lambda x: float(x[1]), reverse=False)
        return sorted_memories[:3]

    def addMemory(
        self, prompt_planner: any, human_question: str, result: any, comments: str = ""
    ):
        prompt_planner = prompt_planner.replace("'", "")
        get_results = self.scenario_memory._collection.get(
            where_document={"$contains": prompt_planner}
        )
        if len(get_results["ids"]) > 0:
            # if already have one, modify it
            id = get_results["ids"][0]
            self.scenario_memory._collection.update(
                ids=id,
                metadatas={
                    "human_question": human_question,
                    "LLM_response": result,
                    "comments": comments,
                },
            )
            print("Modify a memory item.")
        else:
            # else add a new item
            doc = Document(
                page_content=prompt_planner,
                metadata={
                    "human_question": human_question,
                    "LLM_response": result,
                    "comments": comments,
                },
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item.")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print(
            "Delete",
            len(ids),
            "memory items. Now the database has ",
            len(
                self.scenario_memory._collection.get(include=["embeddings"])[
                    "embeddings"
                ]
            ),
            " items.",
        )

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        current_documents = self.scenario_memory._collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        for i in range(0, len(other_documents["embeddings"])):
            if other_documents["embeddings"][i] in current_documents["embeddings"]:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents["embeddings"][i],
                    metadatas=other_documents["metadatas"][i],
                    documents=other_documents["documents"][i],
                    ids=other_documents["ids"][i],
                )
        print(
            "Merge complete. Now the database has ",
            len(
                self.scenario_memory._collection.get(include=["embeddings"])[
                    "embeddings"
                ]
            ),
            " items.",
        )


if __name__ == "__main__":
    pass
