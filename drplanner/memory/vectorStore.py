import os
import textwrap
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document


class DrivingMemory:

    def __init__(self,  db_path=None) -> None:
        self.embedding = OpenAIEmbeddings()            
        db_path = os.path.join(
            './db', 'chroma_5_shot_20_mem/') if db_path is None else db_path
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
            )
        
    def retriveMemory(self, promptPlanner:str, top_k: int = 5):
        similarity_results = self.scenario_memory.similarity_search_with_score(
            promptPlanner, k=top_k)
        fewshot_results = []
        for idx in range(0, len(similarity_results)):
            fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def addMemory(self, prompt_planner: any, corrected_response: str, comments: str = ""):
        prompt_planner = prompt_planner.replace("'", '')
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": prompt_planner
            }
        )

        if len(get_results['ids']) > 0:
            # already have one
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={'LLM_response': corrected_response,  'comments': comments}
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=prompt_planner,
                metadata={'LLM_response': corrected_response,  'comments': comments}
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print("Delete", len(ids), "memory items. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print("Merge complete. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")


if __name__ == "__main__":
    pass