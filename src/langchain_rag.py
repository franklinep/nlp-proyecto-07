import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import Retriever

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No se encontró la GOOGLE_API_KEY en el archivo .env")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

prompt_template_str = """
Responde la pregunta basándote únicamente en el siguiente contexto:

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""
prompt = ChatPromptTemplate.from_template(prompt_template_str)

chain = prompt | llm | StrOutputParser()

retriever = Retriever()
query = "¿Qué es la ansiedad y cómo se relaciona con la búsqueda de empleo?"
print(f"Pregunta del usuario: {query}\n")

print("Recuperando contexto...")
contexto = retriever.get_context(query)
print("Contexto recuperado.\n")

print("Generando respuesta con LangChain...")
inputs = {"context": contexto, "question": query}

respuesta = chain.invoke(inputs)

print("--- Respuesta Final ---")
print(respuesta)
