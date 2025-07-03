from retriever import Retriever
from llm_generator import configure_llm, generate_test_answer
from prompt_manager import create_rag_prompt_template, format_prompt

def run_rag_pipeline(question, top_k=5):
    """
    Ejecuta el pipeline completo de RAG.
    
    Args:
        question (str): La pregunta del usuario.
        top_k (int): El número de casos a recuperar como contexto.

    Returns:
        str: La respuesta final generada por el LLM.
    """
    print("--- INICIANDO PIPELINE RAG ---")

    # 1. RETRIEVE
    print(f"1. Buscando los {top_k} casos más relevantes para: '{question}'")
    retriever = Retriever(
        index_path="../models/patient_cases.index",
        cases_path="../models/patient_cases.pkl",
        model_name="all-mpnet-base-v2"
    )
    retrieved_cases = retriever.search(query=question, top_k=top_k)
    
    contexto = "\n\n---\n\n".join(retrieved_cases)
    print("2. Contexto recuperado.")

    # 2. FORMAT PROMPT
    prompt_template = create_rag_prompt_template()
    final_prompt = format_prompt(template=prompt_template, contexto=contexto, pregunta=question)
    print("3. Prompt final generado.")

    # 3. GENERATE
    print("4. Enviando prompt a Gemini para generar la respuesta...")
    final_answer = generate_test_answer(final_prompt)

    print("--- PIPELINE FINALIZADO ---")
    return final_answer

if __name__ == "__main__":
    configure_llm()
    
    user_question = "¿Qué tratamiento se le puede dar a una paciente que no tiene ganas de hacer nada en la universidad?"
    
    answer = run_rag_pipeline(user_question)
    
    print("\n========= RESPUESTA DEL ASISTENTE =========\n")
    print(answer)
    print("\n===========================================\n")