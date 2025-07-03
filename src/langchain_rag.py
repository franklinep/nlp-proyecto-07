from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import Retriever
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
prompt_template_str = """
Responde la pregunta basándote únicamente en el siguiente contexto:

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""
prompt = ChatPromptTemplate.from_template(prompt_template_str)
chain = prompt | llm | StrOutputParser()

# flujo RAG
retriever = Retriever(
        index_path="./models/patient_cases.index",
        cases_path="./models/patient_cases.pkl",
        model_name="all-mpnet-base-v2")

query = "Que tratamiento es mejor para pacientes con insomnio por sentir una presión constante por rendir al máximo y miedo a cometer errores??"
contexto = retriever.search(query, top_k=10)
inputs = {"context": contexto, "question": query}
respuesta = chain.invoke(inputs)
print(respuesta)
