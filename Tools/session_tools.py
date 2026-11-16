from dataclasses import dataclass
from typing import Dict, Any, List, Literal, Optional

from Tools.transit_tools import unify_value_format


# =========================
# Public result structures
# =========================
@dataclass
class Answer:
    text: str
    token: str
    facts: Optional[Dict[str, Any]] = None
    cursor: Optional[Dict[str, Any]] = None
    # `used_info` is a flexible "sources" container.
    # Personal pipeline typically emits List[str], while the integrated
    # General pipeline emits a list of citations (dicts with text/url).
    used_info: Optional[List[Any]] = None
    pending: bool = False
    need: Optional[Dict[str, Any]] = None


# ==================================
# Follow-up request (InputNeeded)
# ==================================

class InputNeeded(Exception):
    """
    Raised when we need extra user input to proceed.

    `source` distinguishes the follow-up nature:
      - "wf_selection_critic"    : clarification for workflow selection
      - "executor"  : parameter/value required during a tool/action step
      - "router"    : clarification for router selection

    No 'resume' blob is carried — resume is driven entirely by the
    persisted per-main cursor retrieved via the session token.
    """

    def __init__(self, prompt_text: str, return_keys: List[str], return_types: List[str],
                 source: Literal["wf_selection_critic", "executor", "router"] = "executor"):
        super().__init__(f"Input Needed: {prompt_text}")
        self.state_token = None
        self.text = prompt_text
        self.return_keys = return_keys
        self.return_types = return_types
        self.source = source

    # def to_dict(self):
    #     dict_to_return = {
    #         "prompt_text": self.text,
    #         "return_keys": list(self.return_keys),
    #         "return_types": list(self.return_types),
    #         "source": self.source,
    #     }
    #     if "user_id" in self.return_keys:
    #         dict_to_return["auth"] = True
    #     return dict_to_return
    def to_dict(self):
        return {
            "prompt_text": self.text,
            "return_keys": list(self.return_keys),
            "return_types": list(self.return_types),
            "source": self.source,
        }


# ============ legacy tool-facing helpers (used by actions/endpoints) ============

def get_user_id(prompt_text: str = "מה מספר הנוסע שלך (מופיע על כרטיס מסלול)?"):
    """
    Raise InputNeeded for a user identifier. Backwards compatible helper
    many actions rely on.
    """
    raise InputNeeded(prompt_text, return_keys=["user_id"], return_types=["str"], source="executor")


def query_user(try_to_infer_from_conv, prompt, description, return_keys, return_types):
    """
    Generic ask helper: raises InputNeeded so the orchestrator will prompt the user
    and resume the paused action step using the persisted cursor.
    """
    raise InputNeeded(prompt, return_keys=return_keys, return_types=return_types, source="executor")


def get_selection_from_user(options, prompt, description, return_keys, return_types):
    """
    Generic ask helper: raises InputNeeded so the orchestrator will prompt the user
    and resume the paused action step using the persisted cursor.
    """
    if not options:
        return None
    formatted_options = format_options(options)
    joined_prompt = prompt + "\n\n" + formatted_options
    raise InputNeeded(joined_prompt, return_keys=return_keys, return_types=return_types, source="executor")


def select_option(options, selected_index):
    if not options or not selected_index:
        return None
    return options[int(selected_index) - 1]


def _format_remark(remark: str) -> str:
    if not remark:
        return ""
    lines = [ln.strip() for ln in str(remark).splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n" + "\n".join([f"   > {ln}" for ln in lines])


def format_options(options):
    """
    Format options for display (Markdown-friendly, more readable).

    Output structure (Markdown):
      1. **Title field**: value
         - **Other field**: value
         > remark
    """
    if not (isinstance(options, list) and options):
        return None

    if isinstance(options[0], dict):
        rendered_items = []
        for idx, option in enumerate(options, start=1):
            title_line = None
            bullet_lines = []
            remark_block = ""

            for _key, option_details in option.items():
                if not (
                        isinstance(option_details, dict)
                        and "value" in option_details
                        and "fieldNameHe" in option_details
                        and option_details.get("value")
                ):
                    continue

                label = str(option_details.get("fieldNameHe")).strip()
                value = unify_value_format(option_details.get("value"))

                # First field becomes the title; the rest become bullets
                if title_line is None:
                    title_line = f"**{label}**: {value}"
                else:
                    bullet_lines.append(f"<ins>{label}</ins>: {value}")

                if option_details.get("remark"):
                    remark_block = _format_remark(option_details.get("remark"))

            if title_line is None:
                title_line = "**אפשרות**: (מידע חסר)"

            item = f"{idx}. {title_line}"
            if bullet_lines:
                item += "\n" + "\n".join([f"   - {ln}" for ln in bullet_lines])
            if remark_block:
                item += remark_block

            rendered_items.append(item)

        # Blank line between options = much nicer/less “lifeless”
        return "\n\n".join(rendered_items)

    # Simple list of strings
    return "\n\n".join([f"{i}. **{opt}**" for i, opt in enumerate(options, start=1)])
