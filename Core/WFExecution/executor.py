from Tools.registry import ToolFactory


def keys_in_dict(keys, dict_to_check):
    # Accept string, list, or dict (mapping of inner->target keys)
    if isinstance(keys, dict):
        # consider it present if ANY mapped target key exists in facts
        return any(v in dict_to_check for v in keys.values())
    if isinstance(keys, list):
        return any(k in dict_to_check for k in keys)
    return keys in dict_to_check


def init_registry(actions_specs, endpoints_specs, external_personal_info_services, debug_active, mock_tools):
    """
    Init the actions and tools registry using the given states
    :param actions_specs: The actions specs
    :param endpoints_specs: The endpoints specs
    :param debug_active: Whether the debug mode is active
    :param mock_tools: The mock tools
    :param external_personal_info_services: external services details
    :return: The tools registry
    """
    tool_registry = ToolFactory.create_tool_registry(
        endpoints=endpoints_specs, external_personal_info_services=external_personal_info_services,
        debug_active=debug_active, mock_tools=mock_tools)
    action_registry = ToolFactory.create_action_registry(tool_registry=tool_registry, actions_definition=actions_specs)
    return action_registry


def make_block_action_def(block_id, block: dict):
    """
    Turn a Block spec into an Action spec that just calls the underlying action.
    """
    return {
        "description": f"Block wrapper for {block_id} -> {block['action']['id']}",
        "steps": [
            {
                "tool_name": block["action"]["id"],  # this is another ACTION; ActionRegistry will expand it
                "params": block["action"].get("params", {}) or {}  # nested_params         # static params ride along
            }
        ],
        "params": {},  # block wrappers have no own params
    }


def lookup_dotted(root, path):
    if not path:
        return None
    cur = root
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _apply_aliases_to_key(key_or_dict, aliases):
    if not aliases:
        return key_or_dict
    if isinstance(key_or_dict, str):
        return aliases.get(key_or_dict, key_or_dict)
    if isinstance(key_or_dict, dict):
        return {k: aliases.get(v, v) for k, v in key_or_dict.items()}
    if isinstance(key_or_dict, list):
        return [aliases.get(x, x) for x in key_or_dict]
    return key_or_dict


def _rewrite_param_refs(param_def, aliases):
    # Rewrites {'arg_type':'param','value': 'old'} -> 'new' using aliases
    if not aliases:
        return param_def
    if isinstance(param_def, dict):
        if param_def.get('arg_type') == 'param':
            old = param_def.get('value')
            return {'arg_type': 'param', 'value': aliases.get(old, old)}
        # handle nested dict-param containers and generic dicts
        if param_def.get('arg_type') == 'dict':
            return {'arg_type': 'dict',
                    'value': {k: _rewrite_param_refs(v, aliases) for k, v in (param_def.get('value') or {}).items()}}
        return {k: _rewrite_param_refs(v, aliases) for k, v in param_def.items()}
    if isinstance(param_def, list):
        return [_rewrite_param_refs(x, aliases) for x in param_def]
    return param_def


class Executor:
    """
    The executor is the main class that handles the execution of the Workflows.
    A.k.a. ActionManager
    Cursor-resumable workflow runner (no parallelism).
    Each workflow node is run as an action (block_{id}) using the same step engine as before.
    """

    def __init__(self, data_specs, external_personal_info_services, debug_active, mock_tools):
        # Register synthetic actions for every block
        block_actions = {}
        for block_id, block in data_specs["blocks"].items():
            block_action_name = f"block_{block_id}"
            block_actions[block_action_name] = make_block_action_def(block_id, block)

        # Init tools registry
        self.action_registry = init_registry(
            actions_specs={**data_specs["actions"], **block_actions},
            endpoints_specs=data_specs["endpoints"],
            external_personal_info_services=external_personal_info_services,
            debug_active=debug_active,
            mock_tools=mock_tools,
        )

        self.constants = data_specs["constants"]

    def run_workflow(self, get_workflows, facts, cursor):
        """Run workflow (single implementation).

        Executor remains synchronous.

        External transit API HTTP tools are async-first (httpx.AsyncClient). When invoked
        from this sync pipeline, awaitables are executed via ToolRegistry.call()
        (run_coro_sync bridge), preserving a single workflow execution engine.
        """
        while True:
            result = self.step(get_workflows, facts, cursor)
            if result.get("done"):
                return result

    def _advance_to_next_block(self, cursor, facts, wf):
        """Advance workflow cursor to the next block after completing the current block."""
        cursor["phase_idx"] = 0
        cursor["active_tool_call_name"] = None

        outs = wf.get_adjacency().get(cursor["block_id"], [])
        next_blocks = self._filter_edges_by_condition(outs, facts)
        for edge in next_blocks:
            self._push_block(cursor, edge["to"])

        next_block = self._pop_next_block(cursor)
        cursor["block_id"] = next_block if next_block is not None else "end"
        return {"facts": facts, "cursor": cursor}

    def _merge_doc_outputs(self, facts, tool_details, aliases) -> None:
        """Merge doc_outputs metadata into facts (applies aliases, preserves best level)."""
        doc_meta = (tool_details.get("metadata") or {}).get("doc_outputs") or {}
        doc_level = (tool_details.get("metadata") or {}).get("doc_outputs_level", 10 ** 9)
        if not doc_meta:
            return

        aliased_doc = {aliases.get(k, k): v for k, v in doc_meta.items()}
        wf_doc = facts.get("wf_doc_output") or {}
        existing_docs = dict((wf_doc.get("doc_outputs") or {}))
        level_map = wf_doc.get("_wf_doc_levels") or {}

        for k, v in aliased_doc.items():
            current_best = level_map.get(k)
            if current_best is None or doc_level < current_best:
                existing_docs[k] = v
                level_map[k] = doc_level

        facts["wf_doc_output"] = {**wf_doc, "doc_outputs": existing_docs, "_wf_doc_levels": level_map}

    def _prepare_tool_step(self, cursor, facts, action_spec, step_index, from_facts: bool):
        """Shared pre-processing for sync/async tool step execution.

        Returns a dict with {tool_details, effective_return_data, resolved_params, step_index}
        or None when the step is satisfied from existing facts (cursor advanced).
        """
        tool_details = self.action_registry.get_step(action_spec["name"], step_index)
        aliases = tool_details.get('aliases') or {}
        effective_return_data = _apply_aliases_to_key(tool_details.get("return_data"), aliases)

        # doc outputs
        self._merge_doc_outputs(facts, tool_details, aliases)

        # If this tool was already called and its output is present, just advance
        if cursor.get("active_tool_call_name") == tool_details["name"] and keys_in_dict(effective_return_data, facts):
            cursor["phase_idx"] += 1
            cursor.pop("active_tool_call_name", None)
            return None

        # load-from-facts path (resumed run with persisted facts)
        if from_facts and keys_in_dict(effective_return_data, facts):
            cursor["phase_idx"] += 1
            return None

        # mark which tool is currently active
        cursor["active_tool_call_name"] = tool_details["name"]

        raw_params = (tool_details.get("params") or {})
        aliased_params = {name: _rewrite_param_refs(param_def, aliases) for name, param_def in raw_params.items()}

        # if the step is an InputNeeded step, make sure return_keys are also aliased
        if isinstance(aliased_params.get("return_keys"), dict) and aliased_params["return_keys"].get(
                "arg_type") == "value":
            values = aliased_params["return_keys"].get("value")
            if isinstance(values, list):
                aliased_params["return_keys"] = {"arg_type": "value", "value": [aliases.get(v, v) for v in values]}

        resolved_params = {name: self._resolve(param_def, facts) for name, param_def in aliased_params.items()}
        return {
            "tool_details": tool_details,
            "effective_return_data": effective_return_data,
            "resolved_params": resolved_params,
            "step_index": step_index,
        }

    @staticmethod
    def _apply_tool_result(facts, effective_return_data, result) -> None:
        """Write tool results to facts using effective_return_data mapping."""
        if isinstance(effective_return_data, dict):
            for original_key, value in (result or {}).items():
                target_key = effective_return_data.get(original_key, original_key)
                facts[target_key] = value
        elif isinstance(effective_return_data, list):
            for i, value in enumerate(result or []):
                if i < len(effective_return_data):
                    facts[effective_return_data[i]] = value
        else:
            facts[effective_return_data] = result

    def _begin(self, wf, facts, cursor):
        cursor["phase"] = "run_block_action"
        cursor["phase_idx"] = 0
        if "block_queue" not in cursor:
            cursor["block_queue"] = wf.start_queue()

        if "block_id" not in cursor:
            next_block = self._pop_next_block(cursor)
            cursor["block_id"] = next_block if next_block is not None else "end"
        return {"facts": facts, "cursor": cursor}

    def step(self, get_workflows, facts, cursor):
        wf_id = cursor.get("workflow_id")
        if not wf_id:
            # should have chosen it already
            return {"done": True, "facts": facts, "cursor": cursor, "note": "No workflow selected"}

        wf = get_workflows(wf_id)
        if not wf:
            raise KeyError(f"Unknown workflow: {wf_id}")
        wf = wf[0]
        # Initialize queue for this workflow
        if cursor.get("phase") == "selected_workflow":
            return self._begin(wf, facts, cursor)

        # If we've reached 'end'
        if cursor.get("block_id") == "end":
            cursor.update({
                "phase": "answer",  # next_phase
                "phase_idx": 0,
                "note": "done - executed workflow"
            })
            return {"done": True, "facts": facts, "cursor": cursor}

        # Run current block as an action using the step engine
        action_spec = {"name": f"block_{cursor['block_id']}", "params": {}}

        return self._handle_action(cursor, facts, action_spec, wf, from_facts=True)

    def _handle_action(self, cursor, facts, action_spec, wf, from_facts=True):
        # Run one (resumable) step; if not finished yet, return immediately with the updated cursor
        idx = cursor["phase_idx"]
        total = self.action_registry.get_total_steps_count(action_spec["name"])
        if idx < total:
            return self._run_tool_step(cursor, facts, action_spec, idx, from_facts=from_facts)
        return self._advance_to_next_block(cursor, facts, wf)

    def _run_tool_step(self, cursor, facts, action_spec, step_index, from_facts=False):
        """
        Run a single expanded step and advance the cursor if the step is satisfied.
        :param cursor: a dict containing facts and cursor
        :param facts: Context memory
        :param action_spec: The spec of the tool
        :param idx: The index of the step within the expanded steps
        :param from_facts: Whether to use the context memory
        :return: A dict containing facts and cursor
        """
        prepared = self._prepare_tool_step(cursor, facts, action_spec, step_index, from_facts=from_facts)
        if prepared is None:
            return {"facts": facts, "cursor": cursor}

        result = self.action_registry.call(
            action_spec["name"],
            action_idx=prepared["step_index"],
            params=prepared["resolved_params"],
        )

        self._apply_tool_result(facts, prepared["effective_return_data"], result)

        cursor["phase_idx"] += 1
        cursor.pop("active_tool_call_name", None)
        return {"facts": facts, "cursor": cursor}

    def _resolve(self, val_dict, facts):
        """
        Resolve a value given a param value dict
        :param val_dict: A dict with "value" and "arg_type"
        :param facts: Context memory
        :return: The resolved value
        """
        # val_dict is a dict with "value" and "arg_type". if arg_type is "value", then the value is taken as is
        # if arg_type is "param", then the value is taken from the context
        if not isinstance(val_dict, dict) or "arg_type" not in val_dict:
            # Already a value (safety)
            return val_dict
        arg_type = val_dict["arg_type"]
        match arg_type:
            case "value":
                return val_dict["value"]
            case "param":
                # Support dot notation for nested dictionary access
                key_parts = str(val_dict["value"]).split('.')
                current = facts
                try:
                    for part in key_parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            # Try list index if part is a number
                            if isinstance(current, (list, tuple)) and part.isdigit():
                                idx = int(part)
                                if -len(current) <= idx < len(current):
                                    current = current[idx]
                                else:
                                    return None
                            else:
                                return None
                    return current
                except (KeyError, IndexError, TypeError, AttributeError):
                    return None
            case "dict":
                resolved = {}
                for key, value in val_dict["value"].items():
                    resolved[key] = self._resolve(value, facts)  # recursive call to resolve nested params
                return resolved
            case "const":
                const_name = val_dict.get("value")
                if const_name in self.constants:
                    return self.constants[const_name]
                return None
            case _:
                raise ValueError(f"Unknown arg type {arg_type}")

    @staticmethod
    def _push_block(cursor, block_id):
        if "block_queue" not in cursor:
            cursor["block_queue"] = []
        cursor["block_queue"].append(block_id)

    @staticmethod
    def _pop_next_block(cursor):
        q = cursor.get("block_queue") or []
        return q.pop(0) if q else None

    @staticmethod
    def _filter_edges_by_condition(edges, facts):
        """
        Keep edges whose condition is satisfied.

        Supports:
        - or_conditions: edge is kept if ANY of them is true.
        - and_conditions: edge is kept only if ALL of them are true.

        If both exist on the same edge:
            keep_edge = any(or_conditions) AND all(and_conditions)

        If neither exists → edge is kept.
        Conditions use dotted paths in 'facts' (e.g., 'response.outcome_bool').
        """

        def _condition_matches(cond):
            raw_val = lookup_dotted(facts, cond.get("on"))
            val = raw_val["value"] if isinstance(raw_val, dict) and "value" in raw_val else raw_val

            if "equals" in cond:
                eq = cond.get("equals")

                # normalize "true"/"false" strings to real booleans
                if isinstance(val, str) and val.lower() in ("true", "false"):
                    val = val.lower() == "true"
                if isinstance(eq, str) and eq.lower() in ("true", "false"):
                    eq = eq.lower() == "true"

                return val == eq

            if "gt" in cond:
                gt = cond.get("gt")
                return val > gt

            if "gte" in cond:
                gte = cond.get("gte")
                return val >= gte

            if "lt" in cond:
                lt = cond.get("lt")
                return val < lt

            if "lte" in cond:
                lte = cond.get("lte")
                return val <= lte

            elif "not_empty" in cond:
                # same semantics as before (truthy and len > 0)
                return bool(val) and len(val) > 0

            elif "empty" in cond:
                return (not val) or len(val) == 0

            elif "not_null" in cond:
                return val is not None

            elif "null" in cond:
                return val is None

            else:
                raise ValueError(f"Unknown condition {cond}")

        kept = []

        for e in edges:
            or_conds = e.get("or_conditions") or []
            and_conds = e.get("and_conditions") or []

            # No conditions at all → keep edge
            if not or_conds and not and_conds:
                kept.append(e)
                continue

            keep_edge = True

            # OR-part: if present, at least one must match
            if or_conds:
                keep_edge = any(_condition_matches(c) for c in or_conds)

            # AND-part: if present, all must match (and only if OR-part didn't already fail)
            if keep_edge and and_conds:
                keep_edge = all(_condition_matches(c) for c in and_conds)

            if keep_edge:
                kept.append(e)

        return kept
