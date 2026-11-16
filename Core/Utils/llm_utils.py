import json
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Tools.transit_tools import check_response_detail_validity, unify_value_format


@dataclass
class LLMOptions:
    model_name: str
    temp: float = 0.1
    max_output_tokens: int = 3000
    top_p: float = 1
    seed: Optional[int] = 0

    def __init__(self, model_name: str = "gemini-2.5-flash", *args, **kwargs):
        self.model_name = model_name
        super().__init__(*args, **kwargs)


def load_prompts(required_keys=()) -> dict:
    """
    Load prompts from the prompts.py module.
    
    Returns:
        dict: A dictionary where keys are prompt type names and values are prompt configurations.
        Each configuration must be a dictionary containing at least:
        - 'instructions': str - System instructions for the LLM
        - 'prompt': str - The prompt template with placeholders
        - 'prompt_params': list[str] - List of required parameter names for the template
        
    Raises:
        ImportError: If the prompts module cannot be imported
        ValueError: If no valid prompt configurations are found or if there's an error loading them
    """
    try:
        from Resources.PromptConfigs.prompts import prompts

        # Get all variables from the prompts module that are dictionaries
        prompt_dicts = {}
        for name, value in prompts.items():
            if name.startswith('_') or not isinstance(value, dict):
                continue

            # Validate the basic structure
            if not all(key in value for key in required_keys):
                print(f"Warning: Prompt configuration '{name}' is missing required keys and will be skipped")
                continue

            prompt_dicts[name] = value

        if not prompt_dicts:
            raise ValueError("No valid prompt configurations found in prompts.py. "
                             "Each prompt must be a dictionary with 'instructions', 'prompt', and 'prompt_params' keys.")

        return prompt_dicts

    except ImportError as e:
        raise ImportError(
            f"Failed to import prompts module. Make sure prompts.py exists in Resources/PromptConfigs/: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading prompts: {e}")


def load_prompt_config(prompt_type: str) -> dict:
    """
    Load and validate a specific prompt configuration by its type name.
    
    Args:
        prompt_type: The name of the prompt configuration to load (e.g., 'personal_qa', 'general_qa')
        
    Returns:
        dict: The validated prompt configuration with all required keys
        
    Raises:
        ValueError: If the prompt type is not found or is missing required keys
    """
    if not prompt_type or not isinstance(prompt_type, str):
        raise ValueError("prompt_type must be a non-empty string")
    required_keys = {
        'instructions': str,
        'prompt': str,
        'prompt_params': list,
        'output_schema': list
    }
    prompts = load_prompts(required_keys)

    config = prompts.get(prompt_type)
    if not config:
        available = ", ".join(sorted(f'"{k}"' for k in prompts.keys()))
        raise ValueError(
            f"Prompt type '{prompt_type}' not found. Available types: {available}"
        )

    # Ensure all required keys are present and non-empty
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Prompt type '{prompt_type}' is missing required key: '{key}'")

        if not isinstance(config[key], expected_type):
            raise ValueError(
                f"Prompt type '{prompt_type}' has invalid type for '{key}'. "
                f"Expected {expected_type.__name__}, got {type(config[key]).__name__}"
            )

        if isinstance(config[key], (str, list)) and not config[key]:
            raise ValueError(f"Prompt type '{prompt_type}' has empty value for required key: '{key}'")

    # Make a copy to avoid modifying the original
    return config.copy()


def format_decimals_in_string(str_to_format, precision=4):
    # Find all decimal numbers not preceded by 'wf '
    matches = re.findall(r'(?<!wf )\b\d+\.\d+\b', str_to_format)

    # Format each number to the desired precision
    formatted_numbers = [f"{float(num):.{precision}f}" for num in matches]

    # Replace original numbers with formatted ones
    for original, formatted in zip(matches, formatted_numbers):
        str_to_format = re.sub(rf'(?<!wf )\b{re.escape(original)}\b', formatted, str_to_format, count=1)

    return str_to_format


def _cosine_rank_one_to_many(query_embed, embeddings: List[List[float]]):
    if not embeddings:
        return np.array([], dtype=int), np.array([], dtype=float)
    sims = cosine_similarity([query_embed], embeddings)[0]
    order = np.argsort(-sims)
    return order, sims


class ResponseType(IntEnum):
    """Defines the type of response data."""
    MAIN = 1
    ADDITIONAL = 2
    EMPTY = 3


def format_custom_response(response):
    main_data = []
    additional_data = []
    try:
        main_data, additional_data = _format_custom_response_rec(response, main_data, additional_data)
    except Exception as e:
        raise ValueError(f"Error processing response: {str(e)}")

    format_response = {}
    if len(main_data) > 0:
        format_response["Primary"] = main_data

    if len(additional_data) > 0:
        format_response["Meta"] = additional_data

    return format_response


def _format_custom_response_rec(response, main_arr, add_arr):
    if not response:
        return main_arr, add_arr

    # response is a list: treat each element as its own mini-response object
    if isinstance(response, list):
        # For lists, we don't put things directly into main_arr/add_arr here.
        # Instead, the *caller* (the parent dict key) will wrap each element
        # as { "Primary": [...], "Meta": [...] }.
        for item in response:
            # We still recurse to process nested values *inside* list elements,
            # but we return those to the caller (see dict branch below).
            main_arr, add_arr = _format_custom_response_rec(item, main_arr, add_arr)
        return main_arr, add_arr

    # Dict: either a field spec (has responseType etc.) or a container of fields
    if isinstance(response, dict):
        for key, details in response.items():

            # --- Case 1: nested list (e.g. a list of trips) ---
            if isinstance(details, list):
                list_value = []
                for element in details:
                    elem_main, elem_add = _format_custom_response_rec(element, [], [])

                    elem_obj = {}
                    if elem_main:
                        elem_obj["Primary"] = elem_main
                    if elem_add:
                        elem_obj["Meta"] = elem_add

                    list_value.append(elem_obj)

                # For the list key itself, we keep the key name (e.g. "result")
                # and attach the list of element-objects.
                if len(list_value) > 0:
                    main_arr.append({"name": key, "value": list_value})

            # --- Case 2: field spec dict (has responseType/fieldNameHe/value) ---
            elif isinstance(details, dict) and "responseType" in details:
                response_type = details.get("responseType", ResponseType.EMPTY)
                if response_type == ResponseType.EMPTY:
                    continue
                if not check_response_detail_validity(details):
                    continue

                hebrew_name = details.get("fieldNameHe", "Unknown")
                value = unify_value_format(details.get("value"))
                element = {"name": hebrew_name, "value": value}

                if details.get("remark"):
                    element["comment"] = details["remark"]

                if response_type == ResponseType.MAIN:
                    main_arr.append(element)
                elif response_type == ResponseType.ADDITIONAL:
                    add_arr.append(element)

            # --- Case 3: nested container dict (rare) ---
            elif isinstance(details, dict):
                # If it's not a field-spec, treat it as another container level
                main_arr, add_arr = _format_custom_response_rec(details, main_arr, add_arr)

        return main_arr, add_arr

    # Any other type – nothing to extract at this level
    return main_arr, add_arr


def format_facts_for_prompt(facts):
    output = {}
    doc_outputs = facts.get("wf_doc_output", {}).get("doc_outputs", {})

    for var_name, var_details in doc_outputs.items():
        if not var_details.get("main_return"):
            continue
        else:
            if var_details.get("type") == "response_dict":
                curr_response = format_custom_response(facts.get(var_name))
                if curr_response:
                    output[var_name] = curr_response  # TODO: Use more descriptive name from doc_outputs
            elif var_details.get("type") == "value":
                output[var_name] = f"{var_details['description']}\nvalue: {var_details['value']}\n"

    # compact_output = to_compact_rows(output)
    return format_facts_dict_for_llm_json(output)


def format_facts_dict_for_llm_json(facts_dict) -> str:
    """
    Take the already-formatted dict and return a string for the LLM:
    - Adds short how-to-read instructions.
    - Wraps the JSON in ```json ... ``` fences.
    """

    # If someone already passed a JSON string, just reuse it
    if isinstance(facts_dict, str):
        try:
            parsed = json.loads(facts_dict)
            json_text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), indent=2)
        except Exception:
            # Not valid JSON, just treat as opaque text
            json_text = facts_dict
    else:
        json_text = json.dumps(facts_dict, ensure_ascii=False, indent=2)

    instructions = (
        "- The content below is a JSON object. Top-level keys are sections. Each section usually has:\n"
        "    - 'Primary': list of main facts\n"
        "    - 'Meta' (optional): list of additional information\n"
        "- Each fact item has:\n"
        "    - 'name': Hebrew field label\n"
        "    - 'value': the field value (can be null, number, string, boolean, or list)\n"
        "    - 'comment' (optional): extra explanation about this value\n"
        "- For fields whose 'value' is a list each list element is an object that itself contains 'Primary' and optional 'Meta' lists of facts.\n"
        "- Treat the JSON as ground-truth factual data. If something is null, it means the value is missing / not specified.\n"
        "\n"
        "JSON DATA STARTS BELOW:\n"
    )

    return f"{instructions}```json\n{json_text}\n```"

# _KV_KEYS = {"name", "value", "comment"}
#
#
# def _is_kv_list(lst) -> bool:
#     """
#     Return True if `lst` is a list of dicts shaped like:
#         {"name": ..., "value": ..., ("comment": ...)?}
#     with no extra keys.
#     """
#     if not isinstance(lst, list) or not lst:
#         return False
#
#     for item in lst:
#         if not isinstance(item, dict):
#             return False
#         if "name" not in item or "value" not in item:
#             return False
#         # no extra keys beyond name/value/comment
#         if any(k not in _KV_KEYS for k in item.keys()):
#             return False
#
#     return True


# def _to_compact(obj):
#     """
#     Recursively convert:
#       - lists of {name,value,comment?} dicts -> list of [name, value] or [name, value, comment]
#       - everything else stays structurally the same, but children are converted.
#     """
#     # Case 1: dict
#     if isinstance(obj, dict):
#         return {k: _to_compact(v) for k, v in obj.items()}
#
#     # Case 2: list
#     if isinstance(obj, list):
#         # If this is a KV list, compress to rows
#         if _is_kv_list(obj):
#             rows = []
#             for item in obj:
#                 name = item.get("name")
#                 value = _to_compact(item.get("value"))
#                 row = [name, value]
#
#                 comment = item.get("comment", None)
#                 # only add comment if it exists and is not None
#                 if comment is not None:
#                     row.append(comment)
#
#                 rows.append(row)
#             return rows
#
#         # Otherwise, just recurse into each element
#         return [_to_compact(x) for x in obj]
#
#     # Case 3: scalar (str, int, float, bool, None, etc.)
#     return obj
#
#
# def to_compact_rows(data):
#     """
#     convert entire response dict to compact row format.
#     """
#     return _to_compact(data)
