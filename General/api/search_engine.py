import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import aiohttp

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams

from General.modules.embed_vertexai import embed_texts  # your existing embedder

import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection settings
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "vector_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres")
}

WEAVIATE_URL = "http://localhost:8080"
GRPC_PORT = 50051
API_KEY = None  # set if your Weaviate instance requires authentication

LIMIT_DEFAULT = 50
RRF_K = 60    # The standard RRF constant

class SearchHit(BaseModel):
    certainty: Optional[float]
    distance: Optional[float]
    ref_id: Optional[int] = None
    title: Optional[str] = None
    content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


def embed_query(text: str) -> List[float]:
    """Embed a single query string using your batch embedder."""
    return embed_texts([text])[0]


def sanitize_class_name(db_name: str, suffix: str) -> str:
    """
    Same logic as your ingestion:
    turns 'absences', 'open-file-new' → 'AbsencesContent', 'OpenFileNewQa', etc.
    """
    parts = re.split(r"[^A-Za-z0-9]", db_name)
    capitalized = "".join(word.capitalize() for word in parts if word)
    return f"{capitalized}{suffix}"


def semantic_weaviate_search(
    db_name: str,
    # mode: Literal["content", "qa"],
    query_text: str,
    limit: int = 5,
    certainty: float = 0.5,
) -> Dict[str, Any]:
    """
    Perform semantic vector search using nearVector in Weaviate v4.
    Returns the raw GraphQL JSON dict from Weaviate.
    """
    print("\n\nINSIDE semantic_weaviate_search\n\n")
    client = WeaviateClient(
            connection_params=ConnectionParams.from_url(
                WEAVIATE_URL,
                GRPC_PORT,
            )
        )
    client.connect()
    
    # Search all weaviate vector db's based on the detected class 
    # modes = ["content", "qa"]
    results = []
    # for mode in modes:
        # decide class + fields based on mode
        # if mode == "content":

    # first, get the content data by semantic search
    class_name = sanitize_class_name(db_name, "Content")
    fields = ["ref_id", "title", "content"]


    vector = embed_query(query_text)
    fields_str = "\n".join(fields)

    gql = f"""
    {{
    Get {{
        {class_name}(
        nearVector: {{
            vector: {json.dumps(vector)},
            certainty: {certainty}
        }},
        limit: {limit}
        ) {{
        {fields_str}
        _additional {{
            certainty
            distance
        }}
        }}
    }}
    }}
    """

    result = client.graphql_raw_query(gql)
    returned_result = {
        "raw": result,
        "className": class_name,
        "fields": fields,
        "query": query_text,
    }

    results.append(returned_result)

    '''
    ref_id_list = []
    if result:
        # if result returned from db - save it to the results list and get the relevant pages ids for the QA
        results.append(returned_result)

        result_items = result.get
        if result_items:
            for key, value in result_items.items():

                # If hits is a list: [{...}, {...}]
                if isinstance(value, list):
                    iterable_items = value
                    for item in iterable_items:
    
                        ref_id = item.get("ref_id", None)
                        print("\n\nref_id:", ref_id)
                        if ref_id:
                            ref_id_list.append(ref_id)

    # If we have Contents results,
    # Get all the QA's that belong to the retrived pages by page id (ref_id)
    if ref_id_list:
        for ref_id in ref_id_list:
            class_name = sanitize_class_name(db_name, "Qa")
            fields = ["ref_id", "question", "answer"]

            fields_str = "\n".join(fields)

            # Get all QA data for this topic
            result = client.graphql_raw_query(f"""
            {{
            Get {{
                {class_name}(
                where: {{
                    path: ["ref_id"],
                    operator: Equal,
                    valueInt: {ref_id}
                }}
                ) {{
                {fields_str}
                _additional {{
                    id
                }}
                }}
            }}
            }}
            """)

            returned_result = {
                "raw": result,
                "className": class_name,
                "fields": fields,
                "query": query_text,
            }
            print('\n\nQA result:' , result)
            results.append(returned_result)
            '''


    '''
            qa_collection = client.collections.get(class_name)

            # Iterate all objects
            for obj in qa_collection.iterator():
                print(obj.uuid)
                print(obj.properties)
                # print(obj.vector)        # only if you want the vector
                print("------")

                returned_result = {
                    "raw": result,
                    "className": class_name,
                    "fields": fields,
                    "query": query_text,
                }
            '''
    # else:
    #     raise ValueError("mode must be 'content' or 'qa'")
        

    '''
        vector = embed_query(query_text)
        fields_str = "\n".join(fields)

        gql = f"""
        {{
        Get {{
            {class_name}(
            nearVector: {{
                vector: {json.dumps(vector)},
                certainty: {certainty}
            }},
            limit: {limit}
            ) {{
            {fields_str}
            _additional {{
                certainty
                distance
            }}
            }}
        }}
        }}
        """

        result = client.graphql_raw_query(gql)
        returned_result = {
            "raw": result,
            "className": class_name,
            "fields": fields,
            "query": query_text,
        }
        '''
    client.close()

    return results


def format_results_for_api(
    raw_result: Dict[str, Any],
    class_name: str,
    fields: List[str],
    query: str,
) -> List[Dict[str, Any]]:
    """
    Turn the raw Weaviate GraphQL result into a clean list of dicts:
    [{ ref_id, title/question, answer/content, certainty, distance }, ...]
    """

    # data = raw_result.get("data", {}).get("Get", {})
    # hits = data.get(class_name, [])

    hits = raw_result.get
    # query = raw_result.get("query", "")

    if not hits:
        print("⚠ No results found")
        
    if isinstance(hits, dict):
        # Iterate key-value pairs if hits is a dict
        print("🔍 Hits returned as a DICTIONARY:")
        print("query: ", query)
        print("hits:", hits)
        for key, value in hits.items():
            print(f"{key}: {value}")
    

    # If hits is a dictionary: {"0": {...}, "1": {...}}
    if isinstance(hits, dict):
        iterable_hits = hits.values()

    # If hits is a list: [{...}, {...}]
    elif isinstance(hits, list):
        iterable_hits = hits

    # Unsupported type → no results
    else:
        return []

    formatted = []
    
    print("iterable_hits:", iterable_hits)
    # for obj in iterable_hits:
    # for key, value in hits.items():
    for key, value in iterable_hits.items():

        if isinstance(value, dict):
            obj = value

            additional = obj.get("_additional", {}) or {}
            print("additional:", additional)
            item = {
                "certainty": additional.get("certainty"),
                "distance": additional.get("distance"),
            }
            print("item:", item)

        # Add requested fields (title/content or question/answer)
        # for f in fields:
        #     print("f:", f)
        #     print(f"obj.get({f}):", obj.get(f))
        #     item[f] = obj.get(f)

        for key, value in obj.items():
            print(f"{key}: {value}")
            item[key] = value

    formatted.append(item)

    return formatted

def build_search_response(
    raw_result: Dict[str, Any],
    class_name: str,
    query: str,
    db_name: str,
    mode: str,
    
) -> List[SearchHit]:

    # Extract the class key (e.g. "AbsencesContent")
    class_key = next(iter(raw_result.keys()))
    records = raw_result[class_key] or []

    hits: List[SearchHit] = []

    for record in records:
        additional = record.get("_additional", {}) or {}

        hit = SearchHit(
            certainty=additional.get("certainty"),
            distance=additional.get("distance"),
            ref_id=record.get("ref_id"),
            title=record.get("title"),
            content=record.get("content"),
            question=record.get("question"),
            answer=record.get("answer"),
        )

        hits.append(hit)

    '''
    return SearchResponse(
        query=query,
        dbName=db_name,
        mode=mode,
        className=class_name,
        results=hits,
    )
    '''
    return hits


# -------------------------------------------------------
# Fetch the list of all collections from Weaviate schema
# -------------------------------------------------------
async def get_classes(session):
    url = f"{WEAVIATE_URL}/v1/schema"

    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Schema error {resp.status}: {await resp.text()}")

        data = await resp.json()
        # print("\n\nWeaviate schema data:", data)
        # Extract collection names
        # return [col["name"] for col in data.get("collections", [])]
        return [col["class"] for col in data.get("classes", [])]

'''
# -------------------------------------------------------
# Async vector search on a single collection
# -------------------------------------------------------
async def search_collection(session, collection, vector, limit=LIMIT_DEFAULT):
    # url = f"{WEAVIATE_URL}/v1/collections/{collection}/objects/search"
    url = f"{WEAVIATE_URL}/v1/objects/search"

    payload = {
        "vector": vector,
        "limit": limit,
        "returnMetadata": ["distance", "certainty"]
    }

    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    async with session.post(url, headers=headers, data=json.dumps(payload)) as resp:
        if resp.status != 200:
            text = await resp.text()
            print(f"Error searching {collection}: {resp.status} - {text}")
            return []

        data = await resp.json()

        # Tag results with collection name for debug
        for obj in data.get("objects", []):
            obj["collection"] = collection

        return data.get("objects", [])
'''

async def search_class(session, class_name, query_text, vector, alpha, limit=LIMIT_DEFAULT):
    """
    Uses POST /v1/graphql
    """
    url = f"{WEAVIATE_URL}/v1/graphql"

    # decide class fields based on mode
    if class_name.endswith("Content"):
        # class_name = sanitize_class_name(db_name, "Content")
        fields = ["ref_id", "title", "content"]
    elif class_name.endswith("Qa"):
        # class_name = sanitize_class_name(db_name, "Qa")
        fields = ["ref_id", "question", "answer"]
    else:
        raise ValueError("mode must be 'content' or 'qa'")
    
    fields_str = "\n".join(fields)

    
    # escaped_query = json.dumps(query_text, ensure_ascii=False)

    '''
    # Hybrid configuration
    if vector is not None:
        hybrid_block = f'hybrid: {{ query: "{query_text}", vector: {json.dumps(vector)}, alpha: {alpha} }}'
    else:
        hybrid_block = f'hybrid: {{ query: "{query_text}", alpha: {alpha} }}'
        # hybrid_block = f'bm25: {{ query: "{json.dumps(query_text, ensure_ascii=False)}" }}'
    
    query = {
        "query": f"""
        {{
            Get {{
                {class_name}(
                    {hybrid_block}
                    limit: {limit}
                ) {{
                    {fields_str}
                    _additional {{
                        distance
                    }}
                }}
            }}
        }}
        """
    }
    '''
    
    query = {
        "query": f"""
        {{
            Get {{
                {class_name}(
                    nearVector: {{ vector: {json.dumps(vector)} }}
                    limit: {limit}
                ) {{
                    {fields_str}
                    _additional {{
                        distance
                    }}
                }}
            }}
        }}
        """
    }
    
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    async with session.post(url, headers=headers, data=json.dumps(query)) as resp:
        if resp.status != 200:
            print(f"❌ {class_name}: {resp.status} - {await resp.text()}")
            return []

        data = await resp.json()
        print(f"✅ {class_name} search response:", data)
        objects = (
            data.get("data", {})
                .get("Get", {})
                .get(class_name, [])
        )

        # attach class name
        for o in objects:
            o["class"] = class_name

        return objects


# -------------------------------------------------------
# 3. APPLY RRF TO MERGE ALL RESULTS
# -------------------------------------------------------
def rrf_fuse(all_results, k=RRF_K):
    """
    Input:  dict[class_name] = [results...]
    Output: list of unified results sorted by RRF score
    """

    scores = {}
    result_objects = {}

    for class_name, results in all_results.items():
        for rank, obj in enumerate(results, start=1):
            # build a stable unique key (uuid is not returned here, so hash properties)
            key = f"{class_name}-{rank}-{json.dumps(obj, sort_keys=True)}"

            # save the result object (only once)
            result_objects[key] = obj

            # add RRF contribution
            score = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0) + score

            # attach score to the object for debugging
            obj["_rrfScore"] = scores[key]

    # sort globally by fused score
    ranked = sorted(result_objects.values(), key=lambda x: x["_rrfScore"], reverse=True)
    return ranked


# -------------------------------------------------------
# Perform auto-detected searches across all collections + RRF FUSION
# -------------------------------------------------------
async def search_all_collections(query_text, limit=LIMIT_DEFAULT):
    vector = embed_query(query_text)
    async with aiohttp.ClientSession() as session:

        # 1. Auto-detect Weaviate classes
        classes = await get_classes(session)
        # classes = ['WelfareRightsQa', 'WelfareRightsContent']  # for testing
        print(f"🔍 Auto-detected classes: {classes}")

        # 2. Create parallel tasks
        tasks = [
            search_class(session, class_name, query_text, vector, alpha=0.25, limit=limit)
            for class_name in classes
        ]

        # 3. Run all searches in parallel
        results_per_class = await asyncio.gather(*tasks)
    '''
    # 4. Flatten results
    merged = [obj for lst in results_per_class for obj in lst]
    print(f"\n\n🔍 Merged results from all collections: {merged}")
    # 5. Sort by distance (lower = closer)
    # merged.sort(key=lambda x: x["metadata"]["distance"])
    merged.sort(key=lambda x: x["_additional"]["distance"])

    return merged
    '''

    # Organize results by class for RRF
    # build map: class_name -> results
    all_results = {
        class_name: results
        for class_name, results in zip(classes, results_per_class)
    }

    # apply RRF fusion
    fused = rrf_fuse(all_results)

    return fused


######################## functions from Maor postgres project ########################

def search_table(query_text: str, topic_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a vector similarity search on a single table.
    
    Returns:
        List of dicts with fields: ref_id, title/content/question, content/answer, distance, score
    """
    # embedder = VertexEmbedder(project_id=os.getenv("GCP_PROJECT_ID"))
    
    # Embed the query
    # query_vector = embedder.embed_texts([query_text])[0]
    query_vector = embed_query(query_text)
    # Connect to Postgres
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()    
    
    # We start by searching the *Content table of the selected Topics,
    # only afterwards we are retriving ALL the QA's of the pages from the vector search
    table_name_content = sanitize_class_name(topic_name, "Content")
    # print('sanitize_class_name: ', table_name_content)

    # Determine if it's a Content or Qa table
    is_qa = table_name_content.endswith("Qa")
    if is_qa:
        columns = "ref_id, question, answer"
    else:
        columns = "ref_id, title, content"

    # Execute vector search
    search_query = f"""
        SELECT {columns}, (embedding <=> %s::vector) AS distance
        FROM "{table_name_content}"
        ORDER BY distance ASC
        LIMIT %s;
    """
    
    results = []
    try:
        # Check if table exists first
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name_content,))
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print(f"   ⚠️  Table {table_name_content} does not exist, skipping...")
            return results
        
        cursor.execute(search_query, (query_vector, limit))
        rows = cursor.fetchall()
        
        # getting all the unique ref_id's from the *Content tables for retriving the relevant QS'a
        ref_ids_list = []
        ref_ids_unique_list = []

        for row in rows:
            ref_ids_list.append(row[0])
            dist = row[-1]
            score = 1 - dist  # Convert distance to similarity score
            
            if is_qa:
                result = {
                    "ref_id": row[0],
                    "question": row[1],
                    "answer": row[2],
                    "distance": dist,
                    "score": score,
                    "type": "qa"
                }
            else:
                result = {
                    "ref_id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "distance": dist,
                    "score": score,
                    "type": "content"
                }
            results.append(result)
        
        ref_ids_unique_list = list(set(ref_ids_list))
        # print("\n\nref_ids_unique_list: ", ref_ids_unique_list)

        table_name_qa = sanitize_class_name(topic_name, "Qa")
        # print('sanitize_class_name_QA: ', table_name_qa)

        for ref_id in ref_ids_unique_list:

            search_query_qa = f"""
                SELECT ref_id, question, answer
                FROM "{table_name_qa}"
                WHERE ref_id = {ref_id};
            """
            cursor.execute(search_query_qa)
            rows = cursor.fetchall()
            # print('qa rows:', rows)

            if rows:
                for row in rows:
                    result = {
                    "ref_id": row[0],
                    "question": row[1],
                    "answer": row[2],
                    # "distance": dist,
                    # "score": score,
                    "type": "qa"
                }
                    results.append(result)

        # all_qa_tables_results = search_all_qa_tables(cur=cursor, query_vec=query_vector)
        # print('\n\all_qa_tables_results: ', all_qa_tables_results)


    except Exception as e:
        print(f"❌ Search failed for table {table_name_content}: {e}")
    finally:
        cursor.close()
        conn.close()

    # print("\n=== Search Results ===")
    # print(results)


    return results


def list_qa_tables(cur, schema='public'):
    cur.execute("""
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname = %s
          AND tablename LIKE %s
        ORDER BY tablename;
    """, (schema, '%Qa'))
    return [r[0] for r in cur.fetchall()]


def search_all_qa_tables(query, top_n=5, per_table_k=50, schema='public'):

    query_vec = embed_query(query)
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    # with psycopg2.connect(**DB_CONFIG) as conn:
    #     with conn.cursor() as cursor:
    try:
        qa_tables = list_qa_tables(cursor, schema=schema)
        if not qa_tables:
            raise RuntimeError("No tables found ending with 'Qa' in schema 'public'")

        # Build: (SELECT ... LIMIT per_table_k) UNION ALL (SELECT ... LIMIT per_table_k) ...
        parts = []
        for t in qa_tables:
            parts.append(sql.SQL("""
                (SELECT {tname} AS source_table,
                        ref_id, question, answer,
                        (embedding <=> %s::vector) AS distance
                    FROM {schema}.{table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT {k})
            """).format(
                tname=sql.Literal(t),
                schema=sql.Identifier(schema),
                table=sql.Identifier(t),
                k=sql.Literal(per_table_k),
            ))

        union_sql = sql.SQL(" UNION ALL ").join(parts)
        final_sql = sql.SQL("""
            WITH candidates AS ({union})
            SELECT source_table, ref_id, question, answer, distance
            FROM candidates
            ORDER BY distance
            LIMIT {top_n};
        """).format(union=union_sql, top_n=sql.Literal(top_n))

        # Each part uses %s twice (distance + order), so params repeat:
        params = []
        for _ in qa_tables:
            params.extend([query_vec, query_vec])

        cursor.execute(final_sql, params)
        results = cursor.fetchall()

    except Exception as e:
        print(f"❌ Search failed for search_all_qa_tables: {e}")
    finally:
        cursor.close()
        conn.close()
    return results