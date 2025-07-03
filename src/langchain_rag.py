import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
from retriever import Retriever

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
prompt_template = PromptTemplate(
    template=prompt_template_str,
    input_variables=["context", "question"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)
# flujo RAG
retriever = Retriever(
        index_path="./models/patient_cases.index",
        cases_path="./models/patient_cases.pkl",
        model_name="all-mpnet-base-v2")

query = "Que tratamiento es mejor para pacientes con insomnio por sentir una presión constante por rendir al máximo y miedo a cometer errores??"
contexto = retriever.search(query, top_k=10)
inputs = {"context": contexto, "question": query}
respuesta = llm_chain.invoke(inputs)
print(respuesta['text'])
