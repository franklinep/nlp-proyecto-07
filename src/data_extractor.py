import docx

def extract_text_from_docx(file_path):
    """
    Extrae el texto de un archivo .docx.

    Args:
        file_path (str): La ruta al archivo .docx.

    Returns:
        str: El texto extraído del documento.
    """
    try:
        document = docx.Document(file_path)
        
        full_text = []
        
        for para in document.paragraphs:
            full_text.append(para.text)
            
        return '\n'.join(full_text)
        
    except FileNotFoundError:
        return "Error: El archivo no se encontró en la ruta especificada."
    except Exception as e:
        return f"Ha ocurrido un error inesperado: {e}"

if __name__ == "__main__":
    file_path = "../data/Casos.docx"
    
    clinical_text = extract_text_from_docx(file_path)
    
    print(clinical_text)