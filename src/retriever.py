import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, index_path, cases_path, model_name):
        print("Cargando Retriever...")
        self.index = faiss.read_index(index_path)
        with open(cases_path, "rb") as f:
            self.patient_cases = pickle.load(f)
        self.model = SentenceTransformer(model_name)
        print("Retriever listo.")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query)
        query_embedding_2d = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding_2d, top_k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.patient_cases[idx])
        return results

if __name__ == "__main__":
    query = "paciente con ansiedad por no encontrar trabajo"
    retriever = Retriever(
        index_path="../models/patient_cases.index",
        cases_path="../models/patient_cases.pkl",
        model_name="all-mpnet-base-v2"
    )
    results = retriever.search(query)

    for result in results:
        print(result)
