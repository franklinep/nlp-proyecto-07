from data_extractor import extract_text_from_docx
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def segment_cases(full_text):
    """
    Segmenta el texto completo en una lista de casos clínicos individuales.

    Args:
        full_text (str): Todo el texto extraído del documento.

    Returns:
        list: Una lista donde cada elemento es el texto de un caso clínico.
    """
    # Patrón para dividir el texto en casos individuales
    # Busca una nueva línea seguida de "Nombre" o "Nombre:" o "Nombre de la paciente" o "Nombre de la paciente:"
    pattern = r'\n(?=Nombre[:\sde\sla\spaciente]*)'

    cases = re.split(pattern, full_text)

    processed_cases = []
    for case in cases:
        stripped_case = case.strip()
        if stripped_case:
            if not stripped_case.startswith("Nombre"):
                processed_cases.append("Nombre" + stripped_case)
            else:
                processed_cases.append(stripped_case)

    return [case for case in processed_cases if case.startswith("Nombre")]

def embedding_encode(arr, model):
    passage_embeddings = model.encode(arr)
    # tensor -> numpy array
    passage_embeddings = np.array(passage_embeddings).astype('float32')  # Convertimos a float32 para FAISS
    return passage_embeddings


if __name__ == "__main__":
    file_path = "./data/Casos.docx"

    clinical_text = extract_text_from_docx(file_path)

    patient_cases = segment_cases(clinical_text)
    model = SentenceTransformer("all-mpnet-base-v2") # cargamos el modelo preentrenado embedding
    patient_cases_embeddings = embedding_encode(patient_cases, model) # salida: shape [30, 786] -> 786 es el tamaño del embedding
    # ahora vamos a crear el iindice faiss, usando la clase IndexFlatL2
    index = faiss.IndexFlatL2(len(patient_cases_embeddings[0]))  # L2 distance
    index.add(patient_cases_embeddings)
