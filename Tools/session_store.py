from typing import Dict, Any, Literal, Optional, Tuple, List
from uuid import uuid4
from datetime import datetime


# ==============
# Session memory
# ==============

class SessionMemory:
    """
    Flat per-thread (conversation) layout:

    store = {
      "user_id": "...",
      "conversation_id": "...",   # == thread_id
      "current_main_id": "<uuid>|None",
      "main_order": [<uuid>, ...],
      "mains": {
        "<main_id>": {
          "question": str,
          "route": "general|personal|blocked|casual|None",
          "history": [ {ts, role, type, text, source?} ],
          "cursor": {},
          "facts": {},            # non-flat facts per main
          "facts_log": [],        # append-only provenance per main
          "pending_input": {      # only when waiting for user
             "prompt_text": str,
             "return_keys": [str],
             "return_types": [str],
             "source": "wf_selection_critic|executor"
          } | None,
          "answer": None | str
        }
      }
    }
    """

    def __init__(self, memory_dict=None):
        self.store: Dict[str, Any] = memory_dict if memory_dict is not None else {}

    def get(self, k: str, default=None): return self.store.get(k, default)

    def set(self, k: str, v: Any): self.store[k] = v

    def update(self, patch: Dict[str, Any]): self.store.update(patch)


class SessionStore:  # TODO: replace with redis or any other session store
    """
    In-memory session index:
      - token -> SessionMemory
      - (user_id, conversation_id) -> token  (conversation == thread)
    """

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._index: Dict[Tuple[str, str], str] = {}

    # ---------- basic token management ----------
    def new(self, seed_dict=None, user_id: Optional[str] = None) -> str:
        memory = SessionMemory(seed_dict or {
            "mains": {},
            "main_order": [],
            "current_main_id": None,
            # When a main is decomposed into sub-queries we resume via current_sub_id
            "current_sub_id": None,
        })
        tok = uuid4().hex
        self._sessions[tok] = memory
        if user_id:
            conv_id = uuid4().hex
            self._index[(user_id, conv_id)] = tok
            memory.set("user_id", user_id)
            memory.set("conversation_id", conv_id)
        now = datetime.utcnow().isoformat()
        memory.set("created_at", now)
        memory.set("updated_at", now)
        return tok

    def get(self, token: str) -> SessionMemory:
        return self._sessions[token]

    # def get_by_user(self, user_id: str, conversation_id: str) -> Optional[str]:
    #     return self._index.get((user_id, conversation_id))

    def set(self, token: str, memory: SessionMemory):
        self._sessions[token] = memory

    def delete(self, token: str):
        mem = self._sessions.pop(token, None)
        if mem:
            uid = mem.get("user_id")
            tid = mem.get("conversation_id")
            if uid and tid:
                self._index.pop((uid, tid), None)

    def touch_updated_at(self, token: str):  # TODO: Maybe move to _append_history_entry
        mem = self.get(token)
        mem.set("updated_at", datetime.utcnow().isoformat())

    # ---------- main (top-level question) lifecycle ----------
    def start_main(self, token: str, question_text: str) -> str:
        """
        Start a top-level conversation
        :param token: a unique token
        :param question_text: str of the question
        :return: main_id
        """
        mem = self.get(token)
        main_id = uuid4().hex
        mains = mem.get("mains") or {}
        mains[main_id] = {
            "question": question_text,
            "route": None,
            "is_submain": False,
            "parent_main_id": None,
            "submain_ids": [],
            "decomposition": None,
            "history": [],
            "cursor": {},
            "facts": {},
            "facts_log": [],
            "pending_input": None,
            "answer": None,
            "used_info": [],
        }
        mem.set("mains", mains)
        order = mem.get("main_order") or []
        order.append(main_id)
        mem.set("main_order", order)
        mem.set("current_main_id", main_id)
        self._append_history_entry(mem, main_id, role="user", mtype="main", text=question_text)
        self.touch_updated_at(token)
        return main_id

    def get_current_main(self, token: str) -> Optional[str]:
        mem = self.get(token)
        return mem.get("current_main_id")

    # ---------- current submain ----------
    def get_current_sub(self, token: str) -> Optional[str]:
        mem = self.get(token)
        return mem.get("current_sub_id")

    def set_current_sub(self, token: str, sub_id: Optional[str]):
        mem = self.get(token)
        mem.set("current_sub_id", sub_id)
        self.touch_updated_at(token)

    # ---------- submains (split sub-queries) ----------
    def start_submain(self, token: str, parent_main_id: str, question_text: str,
                      route: Optional[Literal["general", "personal"]] = None, idx: Optional[int] = None,
                      intro: Optional[str] = None) -> str:
        """Create a submain under an existing parent main.

        Submains are not added to main_order, so the UI still shows only top-level mains.
        We mirror their history entries into the parent main history.
        """
        mem = self.get(token)
        mains = mem.get("mains") or {}
        if parent_main_id not in mains:
            raise KeyError(f"parent_main_id {parent_main_id!r} not found")

        sub_id = uuid4().hex
        mains[sub_id] = {
            "question": question_text,
            "route": route,
            "is_submain": True,
            "parent_main_id": parent_main_id,
            "sub_idx": idx,
            "sub_intro": intro,
            "history": [],
            "cursor": {},
            "facts": {},
            "facts_log": [],
            "pending_input": None,
            "answer": None,
            "used_info": [],
        }

        parent = mains[parent_main_id]
        parent.setdefault("submain_ids", [])
        parent["submain_ids"].append(sub_id)

        mem.set("mains", mains)

        # record the sub-question's initial user message
        self._append_history_entry(mem, sub_id, role="user", mtype="main", text=question_text)
        self.touch_updated_at(token)
        return sub_id

    def list_submains(self, token: str, parent_main_id: str) -> List[Dict[str, Any]]:
        mem = self.get(token)
        mains = mem.get("mains") or {}
        parent = mains.get(parent_main_id)
        if not parent:
            raise KeyError(f"parent_main_id {parent_main_id!r} not found")

        ids = parent.get("submain_ids") or []
        out: List[Dict[str, Any]] = []
        for sid in ids:
            sm = mains.get(sid) or {}
            out.append({
                "id": sid,
                "idx": sm.get("sub_idx"),
                "intro": sm.get("sub_intro"),
                "question": sm.get("question"),
                "route": sm.get("route"),
                "pending_input": sm.get("pending_input"),
                "answer": sm.get("answer"),
                "used_info": sm.get("used_info"),
            })
        return out

    def resolve_parent_main_id(self, token: str, main_id: str) -> str:
        mem = self.get(token)
        m = (mem.get("mains") or {}).get(main_id) or {}
        parent = m.get("parent_main_id")
        return parent or main_id

    def set_route(self, token: str, main_id: str, route: str):
        mem = self.get(token)
        mem.get("mains")[main_id]["route"] = route

    # ---------- history & pending input ----------
    def append_followup(self, token: str, main_id: str, text: str, source: str | None = None):
        mem = self.get(token)
        self._append_history_entry(mem, main_id, role="user", mtype="followup", text=str(text), source=source)
        self.touch_updated_at(token)

    def append_system(self, token: str, main_id: str, text: str, source: str | None = None):
        mem = self.get(token)
        self._append_history_entry(mem, main_id, role="assistant", mtype="system", text=str(text), source=source)
        self.touch_updated_at(token)

    def append_answer(self, token: str, main_id: str, text: str, used_info: List[Any] | None = None,
                      source: str | None = None):
        mem = self.get(token)
        src = source or "assistant"
        self._append_history_entry(mem, main_id, role="assistant", mtype="answer", text=text, source=src)
        mem.get("mains")[main_id]["answer"] = text
        if used_info is not None:
            mem.get("mains")[main_id]["used_info"] = list(used_info)
        self.touch_updated_at(token)

    def set_pending_input(self, token: str, main_id: str, pending: Dict[str, Any], log_prompt: bool = True):
        mem = self.get(token)
        mem.get("mains")[main_id]["pending_input"] = pending

        # log the system/assistant prompt that requested this input
        if log_prompt:
            prompt_text = pending.get("prompt_text")
            source = pending.get("source")
            if prompt_text:
                self._append_history_entry(mem, main_id, role="assistant", mtype="system", text=prompt_text,
                                           source=source)
        self.touch_updated_at(token)

    def pop_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        mem = self.get(token)
        p = mem.get("mains")[main_id].get("pending_input", None)
        mem.get("mains")[main_id]["pending_input"] = None
        self.touch_updated_at(token)
        return p

    def get_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        mem = self.get(token)
        return mem.get("mains")[main_id].get("pending_input")

    def get_question_text(self, token: str, main_id: str) -> str:
        mem = self.get(token)
        m = (mem.get("mains") or {}).get(main_id) or {}
        return m.get("question") or ""

    # ---------- facts (per main) ----------
    def get_facts(self, token: str, main_id: str) -> Dict[str, Any]:
        mem = self.get(token)
        return mem.get("mains")[main_id]["facts"]

    def set_facts(self, token: str, main_id: str, facts: Dict[str, Any]):
        mem = self.get(token)
        mem.get("mains")[main_id]["facts"] = facts
        self.touch_updated_at(token)

    def log_fact(self, token: str, main_id: str, key: str, value: Any, tool: str, return_data: Any):
        mem = self.get(token)
        mem.get("mains")[main_id]["facts_log"].append({
            "ts": datetime.utcnow().isoformat(),
            "key": key, "value": value, "tool": tool, "return_data": return_data
        })
        self.touch_updated_at(token)

    # ---------- cursor (per main) ----------
    def get_cursor(self, token: str, main_id: str) -> Dict[str, Any]:
        mem = self.get(token)
        return mem.get("mains")[main_id]["cursor"]

    def set_cursor(self, token: str, main_id: str, cursor: Dict[str, Any]):
        mem = self.get(token)
        mem.get("mains")[main_id]["cursor"] = cursor
        self.touch_updated_at(token)

    def list_conversations_for_user(self, user_id: str):
        items = []
        for (uid, conv_id), tok in self._index.items():
            if uid != user_id:
                continue
            mem = self._sessions.get(tok)
            title = self._infer_title(mem)
            items.append({
                "token": tok,
                "title": title,
                "updated_at": mem.get("updated_at"),
            })
        items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return items

    @staticmethod
    def _infer_title(mem: SessionMemory) -> str:
        order = mem.get("main_order") or []
        if not order:
            return "Conversation"
        first = mem.get("mains")[order[0]]
        t = first.get("question") or "Conversation"
        return (t[:60] + "…") if len(t) > 60 else t

    @staticmethod
    def _append_history_entry(mem: SessionMemory, main_id: str, role: str, mtype: str, text: str,
                              source: Optional[str] = None):
        entry = {"ts": datetime.utcnow().isoformat(), "role": role, "type": mtype, "text": text}

        if source:
            entry["source"] = source
        mains = mem.get("mains") or {}
        (mains[main_id]["history"]).append(entry)

        # If this is a submain, mirror the entry into the parent main's history
        m = mains.get(main_id) or {}
        # We intentionally do NOT mirror submain answers into the parent main history.
        # The parent should display only the final composed answer; submain answers remain on the submain.
        if m.get("is_submain") and m.get("parent_main_id") and m.get("parent_main_id") in mains and mtype in (
                "followup", "system", "main"):
            parent_id = m.get("parent_main_id")
            mirrored = dict(entry)
            mirrored["sub_id"] = main_id
            (mains[parent_id]["history"]).append(mirrored)

    def build_conv_history(self, token: str, main_id: str):
        """
        Build a conversation history for a single main, suitable for the UI messages panel.
        Includes the main question, user follow-ups, and assistant answers.
        """
        mem = self.get(token)
        m = mem.get("mains")[main_id]
        msgs = [{"role": "user", "text": m["question"], "type": "main", "source": "user"}]
        for h in m["history"]:
            if h["type"] == "followup":
                msgs.append({
                    "role": "user",
                    "text": h["text"],
                    "type": "followup",
                    "source": h.get("source"),
                })
            elif h["type"] in ("answer", "system"):
                msgs.append({
                    "role": "assistant",
                    "text": h["text"],
                    "type": h["type"],
                    "source": h.get("source"),
                })
        return msgs

    def build_bounded_conv_history(self, token: str, main_id: str, n_last_mains: int = 1,
                                   latest_query_text: Optional[str] = None):
        """
        Bounded contextual memory for LLMs across mains.
        Returns a flat list of messages for the last N mains (including current),
        each expanded via build_conv_history in chronological order.

        If latest_query_text is provided, and the last message in the result is a
        user message with the same text, it is dropped to avoid duplicating the
        current query (which is passed separately to the LLM).
        """
        mem = self.get(token)
        order = mem.get("main_order") or []
        if main_id not in order:
            # Fallback to current main-only formatting
            return self.build_conv_history(token, main_id)

        idx = order.index(main_id)
        n = max(1, int(n_last_mains))
        start = max(0, idx - n + 1)
        selected = order[start:idx + 1]  # inclusive of current

        out: list[dict] = []
        for mid in selected:
            out.extend(self.build_conv_history(token, mid))

        if latest_query_text and out and out[-1].get("role") == "user" and out[-1].get("text") == latest_query_text:
            out = out[:-1]

        return out

    def get_conversation_history(self, token: str):
        """
        Return a flat history from all mains suitable for the UI messages panel:
        [{role: 'user'|'assistant', text: str}, ...]
        """
        mem = self.get(token)
        items: list[dict] = []
        order = mem.get("main_order") or []
        for mid in order:
            items.extend(self.build_conv_history(token, mid))
        return items

    def get_conversation_history_old(self, token: str):
        """
        Return a flat history from all mains suitable for the UI messages panel:
        [{role: 'user'|'assistant'|'system', text: str}, ...]
        """
        mem = self.get(token)
        items = []
        order = mem.get("main_order") or []
        for mid in order:
            main = mem.get("mains")[mid]
            # Mark the start of a main, including the question once
            items.append({"role": "system", "text": f"— שאלה חדשה —\n{main['question']}"})
            for h in main["history"]:
                # Skip the stored "main" entry to avoid duplicating the question
                if h.get("type") == "main":
                    continue
                items.append({"role": h["role"], "text": h["text"]})
        return items
