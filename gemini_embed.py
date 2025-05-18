import google.generativeai as genai

# Set your API key from Google AI Studio
GOOGLE_API_KEY = "AIzaSyBgNMUchYALOtSlngwOQq2Mw8_-DCrj9DE"
genai.configure(api_key=GOOGLE_API_KEY)

# Model ID for embedding
EMBED_MODEL_ID = "models/text-embedding-004"

def gemini_embed(text: str) -> list:
    """
    Returns the embedding vector from Gemini's embedding model.
    """
    response = genai.embed_content(
        model=EMBED_MODEL_ID,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response["embedding"]
