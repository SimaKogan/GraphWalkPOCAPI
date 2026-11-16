# import weaviate
from weaviate import WeaviateClient
from weaviate.classes.config import Configure, Property, DataType
from .embed_vertexai import embed_texts
from tqdm import tqdm
import re


def sanitize_class_name(name: str, mode: str) -> str:
    """
    Sanitize a class name to be GraphQL-safe: PascalCase, no underscores/dashes.
    Appends a suffix like 'Content' or 'Qa'.
    """

    assert mode in ("content", "qa"), "Mode must be 'content' or 'qa'"

    parts = re.split(r"[^A-Za-z0-9]", name)
    capitalized = ''.join(word.capitalize() for word in parts if word)

    full_class_name = f"{capitalized}Content" if mode == "content" else f"{capitalized}Qa"
    return full_class_name


def create_class(client, class_name: str, mode: str) -> None:
    """
    Create a Weaviate collection (class) that accepts user-provided vectors.
    This version is tested on weaviate-client v4.17.0.

    :param client: Connected Weaviate client instance.
    :param class_name: Name of the collection to create.
    """    

    # Sanitize the name to match Weaviate's naming rules
    safe_name = sanitize_class_name(class_name, mode)
    
    # Delete if exists
    # try:
    #     client.collections.delete(safe_name)
    # except Exception:
    #     pass

    if mode == "content":
        props = [
        Property(name="ref_id", data_type=DataType.INT),
        Property(name="table", data_type=DataType.TEXT),
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        ]
    elif mode == "qa":
        props = [
        Property(name="ref_id", data_type=DataType.INT),
        Property(name="table", data_type=DataType.TEXT),
        Property(name="question", data_type=DataType.TEXT),
        Property(name="answer", data_type=DataType.TEXT),
        ]    

    client.collections.create(
        name=safe_name,
        vector_config=Configure.Vectors.self_provided(),
        properties=props,
        )


    print(f"✅ Created class: {safe_name}")
    return safe_name

'''
def build_objects(db_entry, class_name):
    """
    Returns a list of tuples (object, vector)
    """
    objects = []
    
    pages = db_entry.get("pages", [])

    for page in pages:
        page_id = page.get("id")
        page_title = page.get("name", "")
        page_table = db_entry.get("dbName", "")
        
        # Extract content from bodyTexts
        for body in page.get("bodyTexts", []):
            title = body.get("title", "")
            content = body.get("content", "")
            text = f"{title}\n{content}".strip()
            if text:
                objects.append({
                    "class": class_name,
                    "properties": {
                        "ref_id": page_id,
                        "table": page_table,
                        "title": title,
                        "content": content,
                        # "title_with_content": f"{title} {content}".strip(),
                        # "question": None,
                        # "answer": None
                    },
                    "text": text
                })

        # Extract Q&A pairs
        for qa in page.get("matchingQaItems", []):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            text = f"Q: {question}\nA: {answer}".strip()
            text = question.strip()
            if text:
                objects.append({
                    "class": class_name,
                    "properties": {
                        "ref_id": page_id,
                        "table": page_table,
                        # "title": None,
                        # "content": None,
                        "question": question,
                        "answer": answer
                    },
                    "text": text
                })

    
    return objects
'''


def ingest_all(client: WeaviateClient, class_name: str, content_data: list, qa_data: list) -> None:
    """
    For a given subject, create two classes and insert content + qa objects.
    """
        
    content_class = create_class(client, class_name, mode="content")
    qa_class = create_class(client, class_name, mode="qa")

    # "table": class_name,

    # Content insertion
    if content_data:
        texts = [f"{row.get('title', '')} {row.get('content', '')}".strip() for row in content_data]
        vectors = embed_texts(texts)
        collection = client.collections.get(content_class)
        with collection.batch.dynamic() as batch:
            for row, vec in tqdm(zip(content_data, vectors), desc=f"Inserting {content_class}", total=len(vectors)):
                batch.add_object(
                    properties={
                        "ref_id": row["id"],
                        "table": class_name,
                        "title": row.get("title"),
                        "content": row.get("content"),
                    },
                    vector=vec
                )


    # QA insertion
    if qa_data:
        texts = [row.get('question', '').strip() for row in qa_data]
        vectors = embed_texts(texts)
        collection = client.collections.get(qa_class)
        with collection.batch.dynamic() as batch:
            for row, vec in tqdm(zip(qa_data, vectors), desc=f"Inserting {qa_class}", total=len(vectors)):
                batch.add_object(
                    properties={
                        "ref_id": row["id"],
                        "table": class_name,
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                    },
                    vector=vec
                )


def load_kb_file(path: str) -> dict:
    """Flatten a knowledge-base JSON file (Resources/KnowledgeBase/maslul_kb.json).

    Returns {db_name: (content_rows, qa_rows)} where the rows match what :func:`ingest_all` expects.
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for db_entry in data["databases"]:
        content_rows, qa_rows = [], []
        for page in db_entry.get("pages", []):
            for body in page.get("bodyTexts", []):
                content_rows.append({"id": page["id"], "title": body.get("title", ""),
                                     "content": body.get("content", "")})
            for qa in page.get("matchingQaItems", []):
                qa_rows.append({"id": page["id"], "question": qa.get("question", ""),
                                "answer": qa.get("answer", "")})
        out[db_entry["dbName"]] = (content_rows, qa_rows)
    return out


if __name__ == "__main__":
    # python -m General.modules.rebuild_weaviate_from_json
    import weaviate
    from General.modules.config import WEAVIATE_URL
    from urllib.parse import urlparse

    u = urlparse(WEAVIATE_URL)
    with weaviate.connect_to_local(host=u.hostname, port=u.port or 8080) as wclient:
        for name, (content, qa) in load_kb_file("Resources/KnowledgeBase/maslul_kb.json").items():
            ingest_all(wclient, name, content, qa)
