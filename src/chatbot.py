# =================================================================
#  1. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# =================================================================
import os
import uuid
from typing import Annotated, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

from retriever import Retriever

load_dotenv()
# =================================================================
#  DEFINICIÓN DEL ESTADO Y COMPONENTES PRINCIPALES
# =================================================================

# El 'Estado' define la estructura de datos que se comparte entre los nodos del grafo.
# Es la memoria de trabajo de nuestra aplicación.
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # guarda el historial de mensajes.
    question: str # almacena la pregunta actual del usuario.
    context: str # almacena el contexto relevante recuperado para la pregunta.

# --- Componentes de LangChain ---

# El modelo de lenguaje que generará las respuestas.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# El sistema de recuperación (Retriever) que busca información relevante.
print("Cargando Retriever...")
retriever = Retriever(
    index_path="./models/patient_cases.index",
    cases_path="./models/patient_cases.pkl",
    model_name="all-mpnet-base-v2"
)
print("Retriever listo.")

prompt_template = ChatPromptTemplate.from_template(
    """Usa el siguiente contexto y el historial de la conversación para responder la pregunta.

Historial de la Conversación:
{history}

Contexto Relevante:
{context}

Pregunta Actual: {question}

Respuesta:
"""
)

# La cadena (chain) que une el prompt, el modelo y el formateador de salida.
rag_chain = prompt_template | llm | StrOutputParser()


# =================================================================
#  DEFINICIÓN DE LOS NODOS DEL GRAFO
# =================================================================

def retrieve_context_node(state: GraphState):
    """
    Nodo 1: Recupera contexto relevante para la pregunta del usuario.
    """
    print("...recuperando contexto...")
    question = state["messages"][-1].content
    context = retriever.search(question)
    return {"question": question, "context": context}


def generate_answer_node(state: GraphState):
    """
    Nodo 2: Genera una respuesta usando la pregunta, el contexto y el historial.
    """
    print("...generando respuesta...")
    question = state["question"]
    context = state["context"]

    # Formatea el historial para pasarlo al prompt.
    history = "\n".join(
        f"{msg.type}: {msg.content}" for msg in state["messages"][:-1]
    )

    # Invoca la cadena RAG con toda la información necesaria.
    response = rag_chain.invoke({
        "context": context,
        "question": question,
        "history": history
    })
    return {"messages": [AIMessage(content=response)]}


# =================================================================
#  CONSTRUCCIÓN Y COMPILACIÓN DEL GRAFO
# =================================================================

# Creamos una instancia del grafo y le asignamos la estructura de nuestro estado.
workflow = StateGraph(GraphState)

# Registramos nuestras funciones como nodos dentro del grafo.
workflow.add_node("retrieve", retrieve_context_node)
workflow.add_node("generate", generate_answer_node)

# Definimos el flujo de ejecución.
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compilamos el grafo y le añadimos un 'checkpointer' con memoria.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# =================================================================
#  EJECUCIÓN DEL CHATBOT INTERACTIVO
# =================================================================

# Generamos un ID único para la conversación.
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print("\n================================================================================")
print(" Asistente RAG con Memoria Iniciado")
print("================================================================================")
print(f" ID de sesión: {thread_id}")
print(" Escribe 'salir' para terminar la conversación.\n")


while True:
    user_input = input(">> ")
    if user_input.lower() in ["salir", "exit"]:
        print("\n================================================================================")
        print(" Sesión Finalizada")
        print("================================================================================\n")
        break

    print("\n================================ Human Message =================================")
    print(user_input)

    # Enviamos la pregunta del usuario al grafo.
    events = app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="values"
    )

    # Procesamos la salida del stream para obtener la respuesta final.
    for event in events:
        final_state = event

    # Imprimimos la última respuesta generada por el asistente.
    ai_response = final_state["messages"][-1].content
    print("\n================================== Ai Message ==================================")
    print(ai_response)
    print("================================================================================\n")
