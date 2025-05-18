import os
import google.generativeai as genai
from query_engine import query_codebase

# --- Config ---
GOOGLE_API_KEY = "AIzaSyBgNMUchYALOtSlngwOQq2Mw8_-DCrj9DE"
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "models/gemini-2.0-flash"

# --- Format Prompt ---
# def build_prompt(context_chunks, question, max_chars=12000):
#     formatted_chunks = []
#     total_len = 0

#     for chunk in context_chunks:
#         snippet = f"File: {chunk['file_path']}\nService: {chunk['service_name']}\n---\n{chunk['content']}\n"
#         if total_len + len(snippet) > max_chars:
#             break
#         formatted_chunks.append(snippet)
#         total_len += len(snippet)

#     context_text = "\n".join(formatted_chunks)

#     prompt = f"""
# You are an expert software engineer working on a large microservices project.

# Based on the following code context, answer the question at the end. Be detailed and clear. If helpful, write code, list steps, trace logic, or explain connections.

# --- Code Context Start ---
# {context_text}
# --- Code Context End ---

# Question: {question}
# Answer:
# """.strip()

#     return prompt

def build_prompt(context_chunks, question, history=[], max_chars=12000):
    formatted_chunks = []
    total_len = 0

    for chunk in context_chunks:
        snippet = f"File: {chunk['file_path']}\nService: {chunk['service_name']}\n---\n{chunk['content']}\n"
        if total_len + len(snippet) > max_chars:
            break
        formatted_chunks.append(snippet)
        total_len += len(snippet)

    context_text = "\n".join(formatted_chunks)

    # Add conversation history (limit to last 3 exchanges)
    history_text = ""
    for q, a in history[-3:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
You are an expert software engineer working on a large microservices project.

{history_text}
Based on the following code context, answer the question at the end. Be detailed and clear. If helpful, write code, list steps, trace logic, or explain connections.

--- Code Context Start ---
{context_text}
--- Code Context End ---

Question: {question}
Answer:
""".strip()

    return prompt


# --- Generate Answer ---
# def answer_question(user_question):
#     chunks = query_codebase(user_question, top_k=5, max_depth=2)
#     prompt = build_prompt(chunks, user_question)

#     model = genai.GenerativeModel(MODEL_NAME)
#     response = model.generate_content(prompt)

#     return response.text.strip()

# # --- CLI Entry ---
# if __name__ == "__main__":
#     query = input("Ask a question about the codebase: ")
#     answer = answer_question(query)

#     print("\n--- Answer ---\n")
#     print(answer)

def answer_question(user_question, history):
    chunks = query_codebase(user_question, top_k=5, max_depth=2)
    prompt = build_prompt(chunks, user_question, history=history)

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)

    return response.text.strip()

if __name__ == "__main__":
    history = []

    print("Ask questions about the codebase. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = answer_question(query, history)
        print("\nAssistant:\n", answer, "\n")

        # Save Q&A to history
        history.append((query, answer))

