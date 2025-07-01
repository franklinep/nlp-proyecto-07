import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, index, patient_cases, model):
        self.index = index
        self.patient_cases = patient_cases
        self.model = model

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query)
        query_embedding_2d = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding_2d, top_k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.patient_cases[idx])
        return results

# lo ponemos a prueba
if __name__ == "__main__":
    # cargamos el índice FAISS, los casos de pacientes y el modelo
    index = faiss.read_index("patient_cases.index")
    with open("patient_cases.pkl", "rb") as f:
        patient_cases = pickle.load(f)
    model = SentenceTransformer("all-mpnet-base-v2")
    # creamos una instancia del Retriever y realizar una búsqueda
    retriever = Retriever(index, patient_cases, model)
    query = "paciente con ansiedad por no encontrar trabajo"
    results = retriever.search(query)

    for result in results:
        print(result)
