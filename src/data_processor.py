from data_extractor import extract_text_from_docx
import re
from sentence_transformers import SentenceTransformer

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
    passage_embeddings = passage_embeddings.tolist()
    return passage_embeddings


if __name__ == "__main__":
    file_path = "./data/Casos.docx"

    clinical_text = extract_text_from_docx(file_path)

    patient_cases = segment_cases(clinical_text)
    model = SentenceTransformer("all-mpnet-base-v2") # cargamos el modelo preentrenado embedding
    patient_cases_embeddings = embedding_encode(patient_cases, model) # obtenemos los embeddings de los casos
    # salida: shape [30, 786] -> 786 es el tamaño del embedding
    print(f"Shape of patient_cases_embeddings: {len(patient_cases_embeddings)} x {len(patient_cases_embeddings[0])}")

    # print(f"Se han encontrado y procesado {len(patient_cases)} casos de pacientes.")
    # print("-" * 20)

    # if len(patient_cases) > 0:
    #   print("\n--- Muestra del Caso 1 ---")
    #   print(patient_cases[0])
    # if len(patient_cases) > 1:
    #   print("\n--- Muestra del Caso 2 ---")
    #   print(patient_cases[1])
