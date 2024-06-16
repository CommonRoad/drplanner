import os
import textwrap
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document


class PlanningMemory:

    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type
        if encode_type == 'sce_encode':
            # 'sce_encode' is deprecated for now.
            raise ValueError("encode_type sce_encode is deprecated for now.")
        elif encode_type == 'sce_language':
            self.embedding = OpenAIEmbeddings(openai_api_key="sk-proj-Mhj0bzpPIn0JPfNQe052T3BlbkFJUMPYjpihwadeT7jqrt2Z")            
        
            db_path = os.path.join(
                './db', 'chroma_24_mem/') if db_path is None else db_path
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
                )
        else:
            raise ValueError(
                "Unknown ENCODE_TYPE: should be sce_encode or sce_language")
        
    def retrieveMemory(self, scenario_description:str, top_k: int = 5):
        similarity_results = self.scenario_memory.similarity_search_with_score(
            scenario_description, k=top_k)
        fewshot_results = []
        for idx in range(0, len(similarity_results)):
            if similarity_results[0][1] < 0.3:
              fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def addMemory(self, scenario_description: any, human_question: str,corrected_response: str, comments: str = ""):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            scenario_description = scenario_description.replace("'", '')
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": scenario_description
            }
        )

        if len(get_results['ids']) > 0:
            # already have one
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={"human_question": human_question,'LLM_response': corrected_response,  'comments': comments}
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=scenario_description,
                metadata={"human_question": human_question,'LLM_response': corrected_response,  'comments': comments}
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