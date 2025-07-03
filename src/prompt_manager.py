def create_rag_prompt_template():
    """
    Define y devuelve la plantilla de prompt para el sistema RAG.
    """
    template = """
    Eres un asistente experto en analizar historiales clínicos. Tu tarea es responder a la pregunta del usuario de forma clara y concisa, basándote únicamente en el contexto proporcionado.

    **Instrucciones estrictas:**
    1.  Basa tu respuesta exclusivamente en el siguiente contexto.
    2.  No utilices ningún conocimiento externo o previo.
    3.  Si la información para responder la pregunta no se encuentra en el contexto, responde exactamente con: "La información no se encuentra en el contexto proporcionado."
    4.  Cita tus respuestas extrayendo frases textuales del contexto cuando sea posible.

    **Contexto:**
    ---
    {contexto}
    ---

    **Pregunta:**
    {pregunta}

    **Respuesta:**
    """
    return template

def format_prompt(template, contexto, pregunta):
    """
    Inserta el contexto y la pregunta en la plantilla de prompt.
    """
    return template.format(contexto=contexto, pregunta=pregunta)

if __name__ == "__main__":
    prompt_template = create_rag_prompt_template()

    ejemplo_contexto = "Nombre de la paciente: M.G.P. Edad: 27 años. Motivo de consulta: 'No tengo ganas de hacer nada en la universidad, siento que estoy estancada.' La paciente atribuye esta desmotivación a su frustración por no conseguir trabajo."
    ejemplo_pregunta = "¿Cuál es el motivo de consulta de la paciente M.G.P.?"

    prompt_final = format_prompt(
        template=prompt_template, 
        contexto=ejemplo_contexto, 
        pregunta=ejemplo_pregunta
    )

    print("--- TEMPLATE DE PROMPT GENERADA ---")
    print(prompt_final)