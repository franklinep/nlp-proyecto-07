from data_extractor import extract_text_from_docx
import re

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

if __name__ == "__main__":
    file_path = "../data/Casos.docx"
    
    clinical_text = extract_text_from_docx(file_path)
    
    patient_cases = segment_cases(clinical_text)
    
    print(f"Se han encontrado y procesado {len(patient_cases)} casos de pacientes.")
    print("-" * 20)
    
    if len(patient_cases) > 0:
      print("\n--- Muestra del Caso 1 ---")
      print(patient_cases[0])
    if len(patient_cases) > 1:
      print("\n--- Muestra del Caso 2 ---")
      print(patient_cases[1])