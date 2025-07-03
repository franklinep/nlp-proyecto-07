import os
from dotenv import load_dotenv

# --- Importaciones de LangChain ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# Importa tu Retriever personalizado
from retriever import Retriever

# --- 1. CONFIGURACIÓN DE COMPONENTES ---
# Cada herramienta puede necesitar sus propios componentes para funcionar.
load_dotenv()

# El LLM que usará la herramienta RAG internamente.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# El sistema de recuperación (Retriever).
retriever = Retriever(
    index_path="./models/patient_cases.index",
    cases_path="./models/patient_cases.pkl",
    model_name="all-mpnet-base-v2"
)

# La cadena (chain) específica para la lógica RAG.
rag_chain = (
    ChatPromptTemplate.from_template(
        """Responde la pregunta basándote únicamente en el siguiente contexto:

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""
    )
    | llm
    | StrOutputParser()
)


# --- 2. DEFINICIÓN DE HERRAMIENTAS ---

@tool
def patient_case_rag_tool(query: str) -> str:
    """
    Útil para responder preguntas sobre casos de pacientes, tratamientos,
    diagnósticos y temas psicológicos documentados. La entrada debe ser
    una pregunta completa sobre una condición o caso.
    """
    print("--- Ejecutando Herramienta RAG ---")
    context = retriever.search(query)
    response = rag_chain.invoke({"context": context, "question": query})
    return response


@tool
def calculator_tool(expression: str) -> str:
    """
    Útil para resolver expresiones matemáticas simples.
    La entrada debe ser una expresión matemática válida (p. ej., '5 * (3 + 1)').
    """
    print(f"--- Ejecutando Herramienta Calculadora con la expresión: {expression} ---")
    try:
        # Nota: eval() es potente pero puede ser inseguro en producción.
        # Para este ejemplo controlado, es suficiente.
        result = eval(expression)
        return f"El resultado de '{expression}' es {result}."
    except Exception as e:
        return f"Error al evaluar la expresión: {e}"

