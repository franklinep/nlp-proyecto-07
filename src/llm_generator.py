import os
import google.generativeai as genai
from dotenv import load_dotenv

def configure_llm():
    """
    Carga la API key desde el archivo .env y configura el modelo generativo.
    """
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("No se encontró la GOOGLE_API_KEY en el archivo .env")
        
    genai.configure(api_key=api_key)

def generate_test_answer(prompt):
    """
    Envía un prompt a Gemini y devuelve la respuesta generada.

    Args:
        prompt (str): La pregunta o instrucción para el modelo.

    Returns:
        str: La respuesta del modelo como texto.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Error al generar la respuesta: {e}"

if __name__ == "__main__":
    configure_llm()
    
    test_prompt = "Explica qué es la Terapia de Aceptación y Compromiso (ACT) en menos de 50 palabras."
    
    print(f"Prompt: '{test_prompt}'")
    print("-" * 20)

    answer = generate_test_answer(test_prompt)

    print("Respuesta:")
    print(answer)