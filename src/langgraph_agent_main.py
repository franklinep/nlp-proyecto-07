# --- Importaciones de librerías y módulos necesarios ---
import os
import uuid
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict

# --- Importaciones específicas para trabajar con LangGraph (estructura tipo grafo) ---
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition  # Herramientas ya preparadas por LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END

# --- Importamos nuestras herramientas personalizadas ---
from rag_tools import patient_case_rag_tool, calculator_tool

# --- 1. CONFIGURACIÓN INICIAL ---

# Carga las variables de entorno desde el archivo .env (por ejemplo, la API key de Google)
load_dotenv()

# Inicializamos el modelo de lenguaje de Google (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 2. DEFINICIÓN DE HERRAMIENTAS Y ESTADO ---

# Lista de herramientas disponibles para el agente
tools = [patient_case_rag_tool, calculator_tool]

# Asociamos las herramientas al modelo, así podrá decidir cuándo usarlas
llm_with_tools = llm.bind_tools(tools)

# Definimos el estado del agente como una lista de mensajes. LangGraph gestionará el historial.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- 3. DEFINICIÓN DE NODOS DEL GRAFO ---

# Nodo principal (agente): decide si responder o usar una herramienta
def agent_node(state: AgentState):
    print("---  Agente pensando... ---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Nodo de herramientas: ejecuta la herramienta que el agente haya elegido
tool_node = ToolNode(tools)  # Nodo predefinido de LangGraph

# --- 4. CONSTRUCCIÓN DEL GRAFO ---

# Creamos el grafo que define el flujo del agente
graph_builder = StateGraph(AgentState)

# Añadimos los nodos (acciones) al grafo
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

# Establecemos que el flujo siempre inicia en el nodo del agente
graph_builder.add_edge(START, "agent")

# Luego del nodo del agente, verificamos si se debe ir a una herramienta o finalizar
graph_builder.add_conditional_edges("agent", tools_condition)

# Si se usó una herramienta, el resultado vuelve al agente para continuar el razonamiento
graph_builder.add_edge("tools", "agent")

# --- 5. COMPILAR Y EJECUTAR LA APLICACIÓN ---

# Guardamos el estado automáticamente en memoria con MemorySaver
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

# --- INTERFAZ INTERACTIVA EN CONSOLA ---

# Generamos un ID único para la sesión del usuario
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# Mensaje de bienvenida
print("\n================================================================================")
print(" Agente Inteligente con Herramientas Iniciado")
print("================================================================================")
print(f" ID de sesión: {thread_id}")
print(" Puedes hacer preguntas sobre casos de pacientes, matemáticas o simplemente conversar.\n")

# Bucle de interacción con el usuario
while True:
    user_input = input(">> ")
    if user_input.lower() in ["salir", "exit"]:
        break

    # Enviamos el mensaje del usuario al grafo del agente
    events = app.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="values"
    )

    # Recolectamos e imprimimos la respuesta final del agente
    for event in events:
        final_state = event

    ai_response = final_state["messages"][-1]
    print("\n================================== Ai Response =================================")
    print(ai_response.content)
    print("================================================================================\n")
