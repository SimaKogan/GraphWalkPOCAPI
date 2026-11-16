import requests
from General.modules.genai_client import get_client
from google.genai import types
# import config_orchestrator_AI
from pydantic import BaseModel
from typing import List


# ----- Schema -----
class ContextReconstructionResult(BaseModel):
    reconstructed_query: str
    reasoning: str
    context_used: List[int]
    notes: str

def context_reconstruction_agent(last_five_messages: list) -> dict:


    """
    Reconstructs the latest user query (the 5th message) into a fully standalone,
    context-aware Hebrew query based on the last 5 conversation messages.

    The function identifies whether the final user message is standalone or a
    follow-up, then uses relevant context from the preceding messages to produce:
      - reconstructed_query (Hebrew, self-contained)
      - reasoning (brief explanation in Hebrew or English)
      - context_used (indices 0–4 of messages used)
      - notes (optional QA info)

    Parameters
    ----------
    last_five_messages : list
        Exactly 5 conversation messages in chronological order. Each contains:
        - isBot: bool
        - messages: [{ "type": "text", "text": "<content>" }]

    Returns
    -------
    dict
        A JSON-compatible dictionary with reconstructed_query, reasoning,
        context_used, and notes.
    """


    prompt = f"""
    You are given the last 5 messages of a conversation between a user and a bot.

    The messages are provided in chronological order (oldest first, newest last).  
    The LAST message (index 4) is ALWAYS the current user query that must be reconstructed.

    Each message object has:
    - isBot: boolean
    - messages: an array of objects like:
      - {{ "type": "text", "text": "<message content>" }}

    Your task:
    - Carefully read ALL 5 messages.
    - Determine whether the last user query depends on previous turns (follow-up, clarification, refinement) or is standalone.
    - Reconstruct the last user query into a fully standalone Hebrew query that does not rely on the previous messages to be understood.
    - Preserve the original intent and meaning exactly. Do NOT add, remove, or change the user's goals.
    - If the user is referring to something implicitly (e.g., "ובכל זאת?", "ומה לגבי זה?", "כמה זה יוצא בסוף?"), replace those with explicit references derived from the context (for example, referring explicitly to the previous topic like "ההבדל בין מנוי חודשי למנוי שנתי").

    Guidance:
    - Consider both user and bot messages when inferring what "this", "that", or "it" refers to.
    - If the bot previously said it had insufficient information and the user now presses for "any difference", reconstruct the query to explicitly mention that they want any partial difference within the same topic.
    - If the last query clearly starts a new topic, treat it as standalone and keep it as-is (or slightly clarified) without pulling in unrelated context.

    You MUST output JSON only, with the fields:
    - reconstructed_query  (Hebrew, 1–3 sentences, fully standalone)
    - reasoning            (short, high-level explanation in English)
    - context_used         (indices of messages you used: integers 0–4)
    - notes                (optional details for QA; may be empty, in Hebrew or English)

    HERE are the last 5 messages of the conversation (as a JSON array):

    {last_five_messages}

    Remember:
    - The last array item is always the current user query that you are reconstructing.
    - Output only the JSON object, with no extra text.
    """

    client = get_client()

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.4,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        response_mime_type="application/json",
        safety_settings=[types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],

        system_instruction=[types.Part.from_text(
            text=(
                """
                You are a Context Reconstruction Agent.

                Your only job is to transform the user's latest query into a fully reconstructed, standalone, context-aware query in Hebrew, based on the last few turns of the conversation.

                You receive exactly 5 messages from the conversation history, in chronological order:
                - The first item is the oldest of these 5 messages.
                - The last item (index 4) is ALWAYS the current user query that must be reconstructed.

                Each message object has:
                - isBot: boolean
                  - true  = message from the bot/assistant
                  - false = message from the user
                - messages: an array of message parts; each part has:
                  - type: "text"
                  - text: the actual message content string

                Your goals:
                1. Decide whether the current user query (last message) is:
                  - a standalone, self-contained query, OR
                  - a follow-up, clarification, or refinement that depends on previous turns.
                2. Use the previous messages (indexes 0–3) ONLY as context to clarify the user's intention.
                3. Produce a reconstructed query in Hebrew that:
                  - is fully standalone (no pronouns like "this", "that", "there", "it" without context),
                  - preserves the original meaning and intent exactly,
                  - does NOT add new information or assumptions that the user did not imply,
                  - is clear and executable as a task for a downstream answering agent without needing the history.
                4. If the query is already standalone and fully clear, you may keep it almost unchanged, but you may lightly rephrase for clarity as long as the meaning remains identical.
                5. You MUST avoid changing the topic or introducing new topics that were not present in the last 5 messages.

                Important constraints:
                - The reconstructed_query MUST be in Hebrew.
                - You MUST base your reconstruction ONLY on the given last 5 messages. Do NOT use external knowledge.
                - You MUST NOT invent facts, constraints, or user goals that are not supported by the text.
                - You MUST explicitly decide which of the 5 messages were relevant to the reconstruction and list their indices.

                Output format:
                You MUST output a single JSON object with the following fields:

                {
                  "reconstructed_query": "<Hebrew query, 1–3 short sentences, fully standalone>",
                  "reasoning": "<short explanation (in Hebrew or English) describing how you used the context and why this reconstruction matches the user's intent>",
                  "context_used": [<list of integer indices between 0 and 4 that you used>],
                  "notes": "<optional extra observations for QA; can be in Hebrew or English; if you have nothing special to add, you may return an empty string>"
                }

                Additional behavior rules:
                - Do NOT output anything outside this JSON object.
                - Do NOT include code fences (no ```json, no ```).
                - Do NOT expose internal chain-of-thought beyond the short, high-level reasoning requested.
                - If the last user message is clearly unrelated to the previous ones (new topic), treat it as standalone and reconstruct it accordingly.
                - If the last user message is ambiguous even with context, keep the reconstruction conservative and mention the ambiguity in the reasoning and notes.
                """
            )
        )]
    )

    try:
        answer = client.models.generate_content(model=model, contents=contents, config=generate_content_config)
        topic = answer.text
        output_tokens = answer.usage_metadata.candidates_token_count
        input_tokens = answer.usage_metadata.prompt_token_count

        return topic, input_tokens, output_tokens

    except Exception as e:
        print(f"Error extracting topic: {e}")
        return "",0,0  # Return an empty list if there's an error


