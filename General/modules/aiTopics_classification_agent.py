# -*- coding: utf-8 -*-
"""Topic classification for the General (knowledge-base) route.

Takes a Hebrew rider question and returns a JSON string::

    {"query": "...", "topics": [{"topic": "<topic_key>", "confidence": 0-5}, ...]}

The topic definitions live in topics_catalog.py - this module only builds the prompt from them.
"""
import json

from google.genai import types

from General.modules.genai_client import DEFAULT_MODEL, get_client
from General.modules.topics_catalog import TOPICS, render_topics

_SAFETY = [types.SafetySetting(category=c, threshold="OFF") for c in (
    "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_HARASSMENT")]

_COMMON_RULES = """
You are a Topic Classification Engine for the rider help center of Maslul, a fictional public-transport
operator. Rider questions are written in Hebrew.

You ONLY classify. Do NOT answer the question, do NOT explain, do NOT invent topics.
Use ONLY the topics defined between BEGIN_TOPICS_DEFINITION and END_TOPICS_DEFINITION.

Matching, in this order of weight:
1. keywords and synonyms (including plural / gender / construct-state variations and Hebrew root relations),
2. the semantic intent of the query compared to the topic description,
3. the implied context (e.g. "חויבתי פעמיים" -> refunds, "ערעור" -> fines, "אזרח ותיק" -> discounts).

Confidence scale: 5 = clearly the topic, 4 = very likely, 3 = plausible, 2 = weak, 1 = very weak, 0 = unrelated.

OUTPUT: return ONLY valid JSON (no code fences, no text outside it) in this exact format:
{
  "query": "<the original query>",
  "topics": [ { "topic": "<topic_key>", "confidence": <0-5> } ]
}
Rules for the list: at most {max_topics} items, best first, only topic keys from the definitions,
an empty list if nothing is relevant.
"""


def _system_prompt(keys, max_topics) -> str:
    return (_COMMON_RULES.replace("{max_topics}", str(max_topics))
            + "\nBEGIN_TOPICS_DEFINITION\n\n" + render_topics(keys) + "\nEND_TOPICS_DEFINITION\n")


def _classify(current_query: str, keys, max_topics: int) -> str:
    user_prompt = (f"Classify the following user query according to the topic definitions.\n"
                   f"Return ONLY the JSON output defined in the system instructions.\n\nQuery: {current_query}\n")
    config = types.GenerateContentConfig(
        temperature=0.1, top_p=0.25, top_k=5, max_output_tokens=2048,
        response_modalities=["TEXT"], response_mime_type="application/json",
        safety_settings=_SAFETY,
        system_instruction=[types.Part.from_text(text=_system_prompt(keys, max_topics))])
    try:
        resp = get_client().models.generate_content(
            model=DEFAULT_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
            config=config)
        return resp.text
    except Exception as e:  # noqa: BLE001
        print(f"Error extracting topic: {e}")
        return json.dumps({"query": current_query, "topics": []}, ensure_ascii=False)


def aiTopics_classification_agent(current_query: str) -> str:
    """Classify a query against ALL topics (1-3 results)."""
    return _classify(current_query, list(TOPICS), max_topics=3)


# --- Pair classifiers: score a query against two commonly confused topics --------------------------
PAIRS = {
    1: ("discounts", "passes"),
    2: ("refunds", "fines"),
    3: ("card", "lost_property"),
    4: ("fares_and_tickets", "network_and_schedules"),
    5: ("accessibility", "card"),
}


def aiTopics_classification_agent_pair(current_query: str, topic_a: str, topic_b: str) -> str:
    """Score the query against exactly two topics, independently (both may be relevant or irrelevant)."""
    return _classify(current_query, [topic_a, topic_b], max_topics=2)


def aiTopics_classification_agent_pair1(current_query: str) -> str:
    return aiTopics_classification_agent_pair(current_query, *PAIRS[1])


def aiTopics_classification_agent_pair2(current_query: str) -> str:
    return aiTopics_classification_agent_pair(current_query, *PAIRS[2])


def aiTopics_classification_agent_pair3(current_query: str) -> str:
    return aiTopics_classification_agent_pair(current_query, *PAIRS[3])


def aiTopics_classification_agent_pair4(current_query: str) -> str:
    return aiTopics_classification_agent_pair(current_query, *PAIRS[4])


def aiTopics_classification_agent_pair5(current_query: str) -> str:
    return aiTopics_classification_agent_pair(current_query, *PAIRS[5])
