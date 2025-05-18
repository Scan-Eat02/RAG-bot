import os
import uuid
import re
from gemini_embed import gemini_embed 
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- Config ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "project_chunks"

EXCLUDE_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", "venv"}

# --- Setup Vector DB ---
client = QdrantClient(url=QDRANT_URL)

def parse_db_functions(file_path):
    """
    Parses JS DB access files to extract all exported functions.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    exported_funcs = set()

    # Match functions like:
    # async function createEvent(...) {...}
    func_defs = re.findall(r'async\s+function\s+(\w+)\s*\(', content)
    exported_funcs.update(func_defs)

    # Also support named exports like:
    # module.exports = { createEvent, getEvent }
    export_obj = re.search(r'module\.exports\s*=\s*\{([^}]+)\}', content)
    if export_obj:
        for func in export_obj.group(1).split(','):
            func_name = func.strip()
            if func_name:
                exported_funcs.add(func_name)

    return list(exported_funcs)

def parse_use_case_links(file_path):
    """
    Parses a use-case file and returns a list of:
    - use_case_function: The actual exported function name
    - injected_object: Object like eventsDb
    - called_methods: Functions like eventsDb.createEvent
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    links = []

    # Detect module.exports = function makeCreateEvent({ eventsDb }) {
    exported_fn = re.search(r'module\.exports\s*=\s*function\s+(\w+)\s*\(\s*\{\s*([^\}]*)\s*\}', content)
    if not exported_fn:
        return links

    use_case_name, injected_params = exported_fn.groups()
    injected_objects = [obj.strip() for obj in injected_params.split(',') if obj.strip()]

    for obj in injected_objects:
        # Look for calls like: eventsDb.createEvent(...)
        calls = re.findall(rf'{obj}\.(\w+)\s*\(', content)
        for method in calls:
            links.append({
                "use_case_function": use_case_name,
                "calls_db_function": f"{obj}.{method}"
            })

    return links

import re

def parse_controller_links(file_path):
    """
    Extracts controller factory function name and injected use-case dependencies.
    Handles:
    - const makeX = ({ injected }) => { ... }
    - function makeX({ injected }) { ... }
    - module.exports = function makeX({ injected }) { ... }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    controller_links = []

    pattern = re.compile(
        r"""(?x)  # enable verbose mode
        (?:
            const\s+(\w+)\s*=\s*\(\s*\{\s*([^}]+?)\s*\}\s*\)\s*=>      # arrow function
        |
            function\s+(\w+)\s*\(\s*\{\s*([^}]+?)\s*\}\s*\)           # regular function
        |
            module\.exports\s*=\s*function\s+(\w+)\s*\(\s*\{\s*([^}]+?)\s*\}\s*\)  # module.exports = function
        )
        """,
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        arrow_fn, arrow_injected, func_fn, func_injected, export_fn, export_injected = match.groups()

        factory_name = arrow_fn or func_fn or export_fn
        injected_block = arrow_injected or func_injected or export_injected

        if factory_name and injected_block:
            injected_fns = [fn.strip() for fn in injected_block.split(',') if fn.strip()]
            for injected_fn in injected_fns:
                controller_links.append({
                    "controller_function": factory_name,
                    "calls_use_case": injected_fn
                })

    return controller_links

def parse_routes_from_file(file_path):
    """
    Parses route definitions in an Express router file and extracts:
    - method (POST, GET, etc.)
    - path (e.g., /create-event)
    - handler (e.g., eventController.createEventAction)
    """
    routes = []
    route_regex = re.compile(
        r'router\.(post|get|put|delete)\s*\(\s*["\'`](.*?)["\'`]\s*,\s*\(?\s*(?:req\s*,\s*res\s*|\{\s*req\s*,\s*res\s*\})?\)?\s*=>\s*([\w\d_.]+)\('
    )

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = route_regex.search(line)
            if match:
                method, path, handler = match.groups()
                routes.append({
                    "method": method.upper(),
                    "path": path,
                    "handler": handler.strip()
                })
    return routes

def setup_collection():
    if COLLECTION_NAME not in client.get_collections().collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

# --- Code Chunker (Basic for JS/TS) ---
def extract_code_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    current_chunk = []
    inside_fn = False

    for line in lines:
        if "function " in line or "=>" in line or line.strip().startswith("class "):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            inside_fn = True

        if inside_fn:
            current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

# --- Determine Service Name from Path ---
def find_service_name(path_parts):
    """
    Find the nearest parent directory that ends with '-service'.
    Handles any depth and is resilient to relative paths.
    """
    for part in reversed(path_parts):
        if part.endswith("-service") or part.endswith("-panel"):
            return part
    return "general"


# --- Embed & Store Chunks ---
def embed_and_store_chunks(file_path, service_name):
    chunks = extract_code_chunks(file_path)
    for chunk in chunks:
        embedding = gemini_embed(chunk)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "content": chunk,
                "file_path": file_path,
                "service_name": service_name,
                "chunk_type": "code"
            }
        )
        client.upsert(collection_name=COLLECTION_NAME, points=[point])

def process_project_dir(base_path):
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            if file.endswith((".js", ".ts", ".jsx", ".tsx")):
                full_path = os.path.join(root, file)
                path_parts = full_path.split(os.path.sep)
                service_name = find_service_name(path_parts)

                # Determine chunk_type
                if "data-access" in path_parts:
                    db_functions = parse_db_functions(full_path)

                    for chunk in extract_code_chunks(full_path):
                        related_chunks = []
                        for func_name in db_functions:
                            if func_name in chunk:
                                related_chunks.append({
                                    "function_name": func_name,
                                    "relation_type": "defines"
                                })

                        embedding = gemini_embed(chunk)
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "content": chunk,
                                "file_path": full_path,
                                "service_name": service_name,
                                "chunk_type": "db-function",
                                "defined_functions": db_functions,
                                "related_chunks": related_chunks
                            }
                        )
                        client.upsert(collection_name=COLLECTION_NAME, points=[point])

                elif "use-cases" in path_parts:
                    use_case_links = parse_use_case_links(full_path)

                    chunks = extract_code_chunks(full_path)
                    for chunk in chunks:
                        related_chunks = []
                        for link in use_case_links:
                            if link["use_case_function"] in chunk:
                                related_chunks.append({
                                    "function_name": link["calls_db_function"],
                                    "relation_type": "calls"
                                })

                        embedding = gemini_embed(chunk)
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "content": chunk,
                                "file_path": full_path,
                                "service_name": service_name,
                                "chunk_type": "use-case",
                                "related_chunks": related_chunks
                            }
                        )
                        client.upsert(collection_name=COLLECTION_NAME, points=[point])
                elif "routes" in path_parts:
                    route_data = parse_routes_from_file(full_path)

                    for route in route_data:
                        related_chunks = [{
                            "function_name": route["handler"],
                            "relation_type": "calls"
                        }]

                        chunk = f"{route['method']} {route['path']} -> {route['handler']}"
                        embedding = gemini_embed(chunk)

                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "content": chunk,
                                "file_path": full_path,
                                "service_name": service_name,
                                "chunk_type": "route",
                                "related_chunks": related_chunks
                            }
                        )
                        client.upsert(collection_name=COLLECTION_NAME, points=[point])
                elif "controllers" in path_parts:
                    controller_links = parse_controller_links(full_path)

                    chunks = extract_code_chunks(full_path)
                    for chunk in chunks:
                        related_chunks = []

                        # Check if the controller function defined here has any link
                        for link in controller_links:
                            if link["controller_function"] in chunk:
                                related_chunks.append({
                                    "function_name": link["calls_use_case"],
                                    "relation_type": "calls"
                                })

                        embedding = gemini_embed(chunk)
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "content": chunk,
                                "file_path": full_path,
                                "service_name": service_name,
                                "chunk_type": "controller",
                                "related_chunks": related_chunks
                            }
                        )
                        client.upsert(collection_name=COLLECTION_NAME, points=[point])
                else:
                    embed_and_store_chunks(full_path, service_name)

# --- Entry ---
if __name__ == "__main__":
    setup_collection()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and (folder.endswith("-service") or folder.endswith("-panel")):
            process_project_dir(folder_path)
