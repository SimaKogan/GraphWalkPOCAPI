
import json
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from General.modules.aiTopics_classification_agent import aiTopics_classification_agent, aiTopics_classification_agent_pair1, aiTopics_classification_agent_pair2, aiTopics_classification_agent_pair3, aiTopics_classification_agent_pair4, aiTopics_classification_agent_pair5
from General.modules.aiTopics_classification_critic_agent import aiTopics_classification_critic_agent
from General.modules.aiTopics_answer_agent import generate_answer
from General.modules.contex_reconstruction_agent import context_reconstruction_agent
from General.api.search_engine import semantic_weaviate_search, search_all_collections, search_table, search_all_qa_tables


import csv

app = FastAPI(
    title="Weaviate Semantic Search API",
    version="1.0.0",
)

# CORS – adjust origins as needed (for df-messenger frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    dbName: str                     # e.g. "absences"
    # mode: Literal["content", "qa"]  # which class type to search
    limit: int = 5
    certainty: float = 0.5


class SearchHit(BaseModel):
    certainty: Optional[float]
    distance: Optional[float]
    ref_id: Optional[int] = None
    title: Optional[str] = None
    content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    dbName: str
    mode: str
    className: str
    results: List[SearchHit]


# @app.post("/search", response_model=SearchResponse)
@app.post("/search")
def search(req: SearchRequest):
    """
    Main semantic search endpoint.
    Use this from df-messenger or any other client.
    """

    '''
    results_bundle = semantic_weaviate_search(
        db_name=req.dbName,
        # mode=req.mode,
        query_text=req.query,
        limit=req.limit,
        certainty=req.certainty,
    )
    '''
    results_bundle = search_table(
        query_text=req.query,
        topic_name=req.dbName,
        limit=req.limit
    )
    # print("results_bundle:", results_bundle)
    return results_bundle


@app.get("/aiTopics_classify")
async def aiTopics_classify(last_five_messages: str):
# async def aiTopics_classify(userQuery: str):
    """
    Classify the user query into a topic.
    Example usage:
    GET /classify?userQuery=כמה עולה להחליף כרטיס אבוד
    returns list of topics.
    """
    print("\n\n\nlast_five_messages obj:", last_five_messages)
    last_five_messages = json.loads(last_five_messages)
    # print(type(last_five_messages))
    # print("\n\n\nlast_five_messages lenght:", len(last_five_messages))
    ''''''
    if len(last_five_messages) > 1:
        # Reconstruct user query only if there's more than one message
        reconstructed_user_query_obj = context_reconstruction_agent(last_five_messages)
        if reconstructed_user_query_obj:
            reconstructed_user_query_obj = json.loads(reconstructed_user_query_obj[0])
        
            print("\n\n\nReconstructed user query obj:", reconstructed_user_query_obj)
            userQuery = reconstructed_user_query_obj["reconstructed_query"]
    else:
        # Single message, the query will be at: last_five_messages[0]messages[0]["text"]
        userQuery = last_five_messages[0]["messages"][0]["text"]
    
    # userQuery = last_five_messages[0]["messages"][0]["text"]

    search_results = []
    classify_results = json.loads(aiTopics_classification_agent(userQuery))
    # print("\n\n\nclassify_result:", classify_results)

    if classify_results and classify_results["topics"]:

        ####################   WITH CRITIC AGENT   ####################
        '''
        critic_results = json.loads(aiTopics_classification_critic_agent(classify_results["query"], classify_results["topics"]))

        filtered_topics = [
            t["topic"] for t in critic_results.get("final_topics", [])
        ]
        print("\n\n filtered_topics:", filtered_topics)
        for topic in filtered_topics:
            # print("\n\n\ntopic:", topic)            
            searchRequest = SearchRequest(
                query=userQuery,
                dbName=topic,
                # mode="content",
                limit=5,
                certainty=0.1,
            )
            search_result = search(searchRequest)
            # print("\n\n\nsearch result:", search_result)
            # return result
            search_results.append(search_result)
        '''
        ####################   WITHOUT CRITIC AGENT   ####################
        # results = json.loads(classify_results)

        MIN_CONFIDENCE = 4
        filtered_topics = [
            t["topic"] for t in classify_results["topics"]
            if t.get("confidence", 0) >= MIN_CONFIDENCE
        ]
        print('filtered_topics: ', filtered_topics)
        # topics_list = classify_results["topics"]
        for topic in filtered_topics:
            # print("\n\n\ntopic:", topic)            
            searchRequest = SearchRequest(
                query=userQuery,
                dbName=topic,
                # mode="content",
                limit=5,
                certainty=0.5,
            )
            search_result = search(searchRequest)
            # print("\n\n\nsearch result:", search_result)
            # return result
            search_results.append(search_result)

        # We also do a vector search across all the QA's tables
        # The reason is because we do only vector search on the query on the CONTENT tables
        # and we retrive all the QA's of the relevant pages
        # so in case the answer is in the QA's we didnt search for,
        # we can add the top 5 semantic relevant QA's to the results bundle for the answering agent
        top_qa_tables_results = search_all_qa_tables(userQuery)
        print('\n\ttop_qa_tables_results: ', top_qa_tables_results)

        if top_qa_tables_results:
            search_results.append(top_qa_tables_results)

        if search_results:
            # return search_results
            generated_answer = generate_answer(userQuery, search_results)
            print("\n\n\nGenerated answer:", generated_answer)
            return generated_answer
        else:
            return {"error": "No search results found"}
    else:
        print("\n=== Empty classification, searching ACROSS ALL COLLECTIONS ===\n")
        print("\n\nuserQuery:", userQuery)
        # if no result from classification, do a general search across all collections
        results = await search_all_collections(userQuery, limit=50)
        print("\n\nresults from general search:", results)
        print("\n=== TOP RRF RESULTS ===")
        for r in results[:20]:
            print(
                # f"[{r['class']}] score={r['_rrfScore']:.6f}  dist={r['_additional']['distance']:.4f}",
                f"[{r['class']}] score={r['_rrfScore']:.6f}",
                json.dumps(r, ensure_ascii=False)
            )
        return results
            # return {"error": "No topics found, classification failed"}


@app.get("/ask")
async def aiTopics_classify_for_eval(userQuery: str):

    search_results = []
    raw = aiTopics_classification_agent(userQuery)

    # print("RAW CLASSIFIER OUTPUT (repr):", repr(raw))
    # print("RAW LENGTH:", len(raw))
    # print("RAW HEAD:", repr(raw[:300]))
    # print("RAW TAIL:", repr(raw[-300:]))

    classify_results = json.loads(raw)
    # print("\n\n\nclassify_result:", classify_results)

    if classify_results and classify_results["topics"]:
        ####################   WITHOUT CRITIC AGENT   ####################
        # results = json.loads(classify_results)

        MIN_CONFIDENCE = 4
        filtered_topics = [
            t["topic"] for t in classify_results["topics"]
            if t.get("confidence", 0) >= MIN_CONFIDENCE
        ]
        # print('filtered_topics: ', filtered_topics)
        # topics_list = classify_results["topics"]
        for topic in filtered_topics:
            # print("\n\n\ntopic:", topic)            
            searchRequest = SearchRequest(
                query=userQuery,
                dbName=topic,
                # mode="content",
                limit=5,
                certainty=0.5,
            )
            search_result = search(searchRequest)
            # print("\n\n\nsearch result:", search_result)
            # return result
            search_results.append(search_result)

        # We also do a vector search across all the QA's tables
        # The reason is because we do only vector search on the query on the CONTENT tables
        # and we retrive all the QA's of the relevant pages
        # so in case the answer is in the QA's we didnt search for,
        # we can add the top 5 semantic relevant QA's to the results bundle for the answering agent
        top_qa_tables_results = search_all_qa_tables(userQuery)
        # print('\n\ttop_qa_tables_results: ', top_qa_tables_results)

        if top_qa_tables_results:
            search_results.append(top_qa_tables_results)

        if search_results:
            # return search_results
            generated_answer = generate_answer(userQuery, search_results)
            # print("\n\n\nGenerated answer:", generated_answer)
            return generated_answer
        else:
            return {"error": "No search results found"}
    else:
        # print("\n=== Empty classification")
        ''''''
        print("\n=== Empty classification, searching ACROSS ALL COLLECTIONS ===\n")
        print("\n\nuserQuery:", userQuery)
        # if no result from classification, do a general search across all collections
        results = await search_all_collections(userQuery, limit=50)
        print("\n\nresults from general search:", results)
        print("\n=== TOP RRF RESULTS ===")
        # for r in results[:20]:
        #     print(
        #         # f"[{r['class']}] score={r['_rrfScore']:.6f}  dist={r['_additional']['distance']:.4f}",
        #         f"[{r['class']}] score={r['_rrfScore']:.6f}",
        #         json.dumps(r, ensure_ascii=False)
        #     )
        return results
        
        # return "Empty classification"

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/run-csv")
async def run_csv():

    # csv_path = "C:/projects/evaluation_results_sample.csv"
    # csv_output_path = "C:/projects/evaluation_results_sample_output.csv"

    csv_path = "C:/projects/results_95_1.csv"
    csv_output_path = "C:/projects/results_95_output_1.csv"

    # rows = []

    print("SCRIPT STARTED")


    # 1) Read all rows
    with open(csv_path, mode='r', encoding="utf-8") as fin, \
        open(csv_output_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames  # keeps column order

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()


        print(fieldnames)
        # rows = list(reader)
        # rows_len = len(rows)
        # print('rows_len: ', rows_len)x 

        # 2) Update existing rows (NO append)
        # for i, row in enumerate(rows):
        for i, row in enumerate(reader, start=1):
            question = row.get('שאלה')
            print(i, question)

            try:
                # Get the answer from the agent
                answer_raw = await aiTopics_classify_for_eval(userQuery=question)
                # answer_json = json.loads(answer_raw)
                answer = answer_raw['answer']
                # print('\nanswer from the agent: ', answer)
                print('\nanswer number: ', i)
                row[fieldnames[-2]] = answer
                row[fieldnames[-1]] = answer_raw

                writer.writerow(row)
                print('\nanswer ', i,  'written to csv')

            except Exception as e:
                print(e)
        # rows.append(row)

            # if i == 2:
            #     break


    # 3) Write back
    # with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(rows)

    # return {"ok": True, "rows": len(rows), "updated_column": col_f}
    return {"ok": True}
