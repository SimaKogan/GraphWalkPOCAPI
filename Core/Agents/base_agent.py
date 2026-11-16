import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from google.genai import types
from Utils.gcloud_utils import init_genai_client
from Core.Utils.llm_utils import LLMOptions
from Utils.async_utils import to_thread


# ---- Agent tool registry (kept minimal and local to agent layer) ----
@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    description: str = ""


class AgentToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._tools[name]

    def as_sdk_tools(self) -> List[Callable[..., Any]]:
        # google-genai currently accepts Python callables as tools
        return [spec.fn for spec in self._tools.values()]


def get_safety_settings():
    return [types.SafetySetting(
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
    )]


class BaseAgent:
    """
    Minimal, reusable base over google-genai.

    - Stateless by default
    - Shared config & system instruction handling
    - Optional per-call tools (automatic function calling)
    """

    def __init__(self, name: str, system_instruction: Optional[str] = None, llm_options: Optional[LLMOptions] = None,
                 tools: Optional[AgentToolRegistry] = None, automatic_function_calling: bool = True):
        self.name = name
        self.system_instruction = system_instruction or f"You are {name}."
        self.llm_options = llm_options or LLMOptions()
        self._tools = tools or AgentToolRegistry()
        self._auto_fc = automatic_function_calling

        self.client = init_genai_client()

    def generate_text(self, user_text: str, context: Optional[str] = None) -> str:
        """
        Single-pass generation (stateless).
        Sends context and user text as separate messages for clarity.
        """
        # Enforce async usage when called from an active event loop.
        # (The underlying SDK call is synchronous and would block the loop.)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "BaseAgent.generate_text() called from async context; use 'await generate_text_async(...)'."
            )

        # System banner for role separation
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=self._sys_banner())])]

        # Add context in its own message if provided
        if context:
            contents.append(types.Content(role="system", parts=[types.Part.from_text(text=f"Context:\n{context}")]))
        # Optional context and user text in one message (keeps the call simple)
        # payload = (f"[Context]\n{context}\n\n" if context else "") + user_text
        # Add user query as a separate message
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))

        cfg = self._build_cfg(automatic=True)
        out = self.client.models.generate_content(model=self.llm_options.model_name, contents=contents, config=cfg)
        return (out.text or "").strip()

    async def generate_text_async(self, user_text: str, context: Optional[str] = None) -> str:
        """Async wrapper around :meth:`generate_text`.

        The Google GenAI SDK call is synchronous, so we offload to a worker thread
        to avoid blocking the event loop.
        """
        return await to_thread(self.generate_text, user_text, context)


    def generate(self, contents: List[types.Content], system_instruction: Optional[str] = None) -> str:
        """
        Low-level: pass your own contents (stateless).
        """
        # Enforce async usage when called from an active event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "BaseAgent.generate() called from async context; use 'await generate_async(...)'."
            )

        if system_instruction:
            self.system_instruction = system_instruction
        cfg = self._build_cfg(automatic=True)
        out = self.client.models.generate_content(model=self.llm_options.model_name, contents=contents,
                                                  config=cfg)  # TODO: Add stream?
        return (out.text or "").strip()


    async def generate_async(self, contents: List[types.Content], system_instruction: Optional[str] = None) -> str:
        """Async wrapper around :meth:`generate` (thread offload)."""
        return await to_thread(self.generate, contents, system_instruction)

    def _build_cfg(self, automatic: bool) -> types.GenerateContentConfig:
        cfg = types.GenerateContentConfig(
            temperature=self.llm_options.temp,
            top_p=self.llm_options.top_p,
            max_output_tokens=self.llm_options.max_output_tokens,
            seed=self.llm_options.seed,
            system_instruction=[types.Part.from_text(text=self.system_instruction)],
            safety_settings=get_safety_settings(),
            tools=self._tools.as_sdk_tools() if self._tools else None,
        )
        if not automatic:
            cfg.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
            cfg.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
        return cfg

    def _sys_banner(self) -> str:
        # Short tag that helps separate the “real” user text from the sys prompt for some models
        return f"[{self.name}] Use the system instruction above to respond appropriately."
