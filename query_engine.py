import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import google.generativeai as genai

# --- Config ---
genai.configure(api_key="AIzaSyBgNMUchYALOtSlngwOQq2Mw8_-DCrj9DE")

EMBED_MODEL = genai.embed_content
EMBED_MODEL_NAME = "models/text-embedding-004"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "project_chunks"

client = QdrantClient(url=QDRANT_URL)

# --- Embedding Wrapper ---
def embed_text(text: str):
    """Use Gemini Embedding model to get vector."""
    response = genai.embed_content(
        model=EMBED_MODEL_NAME,
        content=text,
        task_type="RETRIEVAL_QUERY"  # or "retrieval_query" depending on context
    )
    return response["embedding"]

# --- Utilities ---
def chunk_id(chunk):
    """Create a hash ID for a chunk for deduplication."""
    base = chunk["file_path"] + "::" + chunk["content"]
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def fetch_related_chunks(initial_chunks, max_depth=2):
    visited = {}
    queue = [(chunk, 0) for chunk in initial_chunks]

    while queue:
        current_chunk, depth = queue.pop(0)

        chunk_key = chunk_id(current_chunk)
        if chunk_key in visited or depth > max_depth:
            continue

        visited[chunk_key] = {
            "chunk": current_chunk,
            "depth": depth,
            "base_score": 1.0 if depth == 0 else 0.75 ** depth  # score decay
        }

        for rel in current_chunk.get("related_chunks", []):
            fn_name = rel["function_name"]
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embed_text(fn_name),
                limit=3,
                with_payload=True
            )

            for match in results:
                rel_chunk = match.payload
                rel_key = chunk_id(rel_chunk)
                if rel_key not in visited:
                    queue.append((rel_chunk, depth + 1))

    return visited

# --- Main Query Function ---
def query_codebase(user_query, top_k=5, service_filter=None, max_depth=2):
    query_embedding = embed_text(user_query)

    query_filter = None
    if service_filter:
        query_filter = Filter(
            must=[
                FieldCondition(key="service_name", match=MatchValue(value=service_filter))
            ]
        )

    initial_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True
    )

    initial_chunks = [res.payload for res in initial_results]
    visited_chunks = fetch_related_chunks(initial_chunks, max_depth=max_depth)

    ranked = []
    for meta in visited_chunks.values():
        content = meta["chunk"]["content"]
        base_score = meta["base_score"]

        ranked.append({
            "file_path": meta["chunk"]["file_path"],
            "service_name": meta["chunk"]["service_name"],
            "chunk_type": meta["chunk"].get("chunk_type"),
            "score": base_score,
            "content": content
        })

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    return ranked

# --- CLI Entry ---
if __name__ == "__main__":
    query = input("Ask a question about the project: ")
    results = query_codebase(query, top_k=5)

    print("\n--- Top Matches with Recursion ---")
    for i, res in enumerate(results[:10]):
        print(f"\n[{i+1}] Score: {res['score']:.4f}")
        print(f"Type: {res['chunk_type']} | File: {res['file_path']} | Service: {res['service_name']}")
        print("Content Preview:\n", res["content"][:300])
