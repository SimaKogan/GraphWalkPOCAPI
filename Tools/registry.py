from typing import Callable, Dict, Any
import inspect

from Utils.async_utils import run_coro_sync

from Tools.MockTools.mock_transit_tools import send_request_stub, send_get_request_stub, send_post_request_stub, \
    format_custom_response_stub, is_date_in_range_stub
from Tools.MockTools.mock_session_tools import get_user_id_stub, query_user_stub, get_selection_from_user_stub
from Tools.transit_tools import send_request_async, send_get_request_async, send_post_request_async, EndpointHandler, \
    is_date_in_range, echo_value, reduce

from Core.Utils.llm_utils import format_custom_response
from Tools.session_tools import get_user_id, query_user, get_selection_from_user, select_option


class ToolRegistry:
    """
    A registry for tools that can be called by name
    """

    def __init__(self):
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]):
        """
        Register a new tool using its name and function
        :param name: A unique name for the tool
        :param fn: The function to register
        """
        self._tools[name] = fn

    def call(self, name: str, **kwargs):
        """Synchronous tool invocation.

        If the underlying tool is async (returns an awaitable), this will execute it
        using :func:`Utils.async_utils.run_coro_sync`.

        This is intended for LocalMode / tests.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        out = self._tools[name](**kwargs)
        if inspect.isawaitable(out):
            return run_coro_sync(out)
        return out

    # NOTE: Async tool invocation (ToolRegistry.acall / ActionRegistry.acall) is intentionally commented out for now.
    #
    # In the current architecture, workflows execute inside a worker thread in both modes:
    #   - LocalMode: workflows are sync and run directly.
    #   - RemoteMode: services offload the entire pipeline via `to_thread` at the HTTP boundary.
    #
    # Therefore the executor uses ActionRegistry.call() -> ToolRegistry.call(), and async-first
    # tools (e.g., httpx-based transit API tools) are handled via "awaitable detection + run_coro_sync".
    #
    # Keep a real `acall` only if/when the workflow executor becomes truly async (runs on an event loop).
    # async def acall(self, name: str, **kwargs):
    #     """Asynchronous tool invocation.
    #
    #     - If the underlying tool is async (coroutine function or returns an awaitable), it is awaited.
    #     - If it is synchronous, it is executed via :func:`Utils.async_utils.to_thread` to avoid blocking the event loop.
    #
    #     Note: We intentionally avoid calling the tool twice (some tools have side-effects).
    #     """
    #     if name not in self._tools:
    #         raise KeyError(f"Unknown tool: {name}")
    #
    #     fn = self._tools[name]
    #
    #     # If this is a coroutine-function, await it directly.
    #     import inspect
    #     if inspect.iscoroutinefunction(fn):
    #         return await fn(**kwargs)
    #
    #     # Otherwise, run in a worker thread. If the function returns an awaitable,
    #     # await it on the event loop (covers wrappers/lambdas that return a coroutine).
    #     out = await to_thread(fn, **kwargs)
    #     if inspect.isawaitable(out):
    #         return await out
    #     return out

    def assert_tool_exists(self, name: str):
        if name not in self._tools:
            return False
        else:
            return True

    @classmethod
    def from_tools(cls, tools):
        """
        Create a ToolRegistry instance from a list of tools
        :param tools: A list of tuples containing the name and function of each tool
        :return: A ToolRegistry instance
        """
        registry = cls()
        for name, fn in tools:
            registry.register(name, fn)
        return registry


class ActionRegistry:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.actions: Dict[str, Dict[str, Any]] = {}
        # self.expanded_actions: Dict[str, List[Dict[str, Any]]] = {}

    def register(self, name: str, action_definition: dict):
        self.actions[name] = action_definition

    def expand_actions(self):
        """
        Expand all registered actions into sequences of tool calls
        """

        for action_name in self.actions:
            self.actions[action_name]['expanded_steps'] = self._expand_action(action_name, set())

    def _resolve_param_with_parent(self, param_info: dict, parent_params: dict = None):
        """
        Recursively resolve a parameter against parent_params, handling nested dicts.
        :param param_info: Parameter info dict with 'value' and 'arg_type'
        :param parent_params: Parent parameters to resolve against
        :return: Resolved parameter dict
        """
        if not isinstance(param_info, dict) or 'arg_type' not in param_info:
            # Safety: if it's not a proper param structure, return as-is
            return param_info

        arg_type = param_info.get('arg_type', 'value')

        if arg_type == 'value':
            # Static value, return as-is
            return {'value': param_info.get('value'), 'arg_type': 'value'}

        elif arg_type == 'param':
            # Parameter reference
            param_name = param_info['value']
            if parent_params and param_name in parent_params:
                parent_param_value = parent_params[param_name]
                # If the parent parameter is a 'value', use its value directly
                if isinstance(parent_param_value, dict) and 'arg_type' in parent_param_value and parent_param_value[
                    'arg_type'] == 'value':
                    return {'value': parent_param_value['value'], 'arg_type': 'value'}
                else:
                    # Otherwise, pass the whole parent parameter structure
                    return parent_param_value
            else:
                # Keep as param reference (to be resolved from facts at runtime)
                return {'value': param_name, 'arg_type': 'param'}

        elif arg_type == 'dict':
            # Recursively process nested dict parameters
            resolved_dict = {}
            for key, nested_param in param_info.get('value', {}).items():
                resolved_dict[key] = self._resolve_param_with_parent(nested_param, parent_params)
            return {'value': resolved_dict, 'arg_type': 'dict'}

        else:
            # Unknown type, return as-is
            return param_info

    def _expand_action(self, action_name: str, visited: set, parent_params: dict = None, depth: int = 0):
        """
        Recursively expand an action into a flat sequence of tool calls
        :param action_name: Name of the action to expand
        :param visited: Set of already visited actions to detect cycles
        :param parent_params: Parameters passed from the parent action
        :return: List of expanded tool calls
        """
        if action_name in visited:
            raise ValueError(f"Circular dependency detected in action: {action_name}")

        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' not found")

        visited.add(action_name)
        action = self.actions[action_name]
        expanded_steps = []

        for step in action.get('steps', []):
            tool_name = step.get('tool_name', '')
            if not tool_name:  # skip empty tool names
                continue

            # Resolve this step's params against the parent (so nested action can inherit call-site values)
            step_params = {}
            for param_name, param_info in step.get('params', {}).items():
                step_params[param_name] = self._resolve_param_with_parent(param_info, parent_params)

            # If this step is actually another action, expand it
            if tool_name in self.actions:
                # Pass the parameters from the current step as parent parameters to the nested action
                nested_steps = self._expand_action(tool_name, visited.copy(), parent_params=step_params,
                                                   depth=depth + 1)

                parent_return = step.get('return_data')
                if nested_steps and parent_return:
                    nested_steps = list(nested_steps)  # avoid mutating cached structures

                    # Attach alias mapping into step metadata; executor will apply at runtime
                    if isinstance(parent_return, str):
                        # map the original last return key -> parent string
                        last_key = nested_steps[-1].get('return_data')
                        if last_key:
                            ns = dict(nested_steps[-1])
                            ns['aliases'] = {**ns.get('aliases', {}), last_key: parent_return}
                            nested_steps[-1] = ns

                    elif isinstance(parent_return, dict):
                        # attach same alias map to all nested steps
                        aliased = []
                        for ns in nested_steps:
                            ns2 = dict(ns)
                            ns2['aliases'] = {**ns2.get('aliases', {}), **parent_return}
                            aliased.append(ns2)
                        nested_steps = aliased

                # Attach the callee action's doc_outputs once (executor will merge them into facts)
                action_def = self.actions.get(tool_name, {})
                action_doc_outputs = action_def.get("doc_outputs") or {}
                if nested_steps and action_doc_outputs:
                    first = dict(nested_steps[0])
                    md = dict(first.get("metadata") or {})
                    md["doc_outputs"] = action_doc_outputs
                    md["doc_outputs_level"] = depth + 1  # <-- lower is more outer; parent is +1
                    first["metadata"] = md
                    nested_steps[0] = first
                expanded_steps.extend(nested_steps)

            else:
                # It's a regular tool call
                expanded_steps.append({
                    'type': 'tool',
                    'name': tool_name,
                    'params': step_params,
                    'return_data': step.get('return_data', []),
                    'metadata': step.get('metadata', {})
                })

        return expanded_steps

    def get_total_steps_count(self, action_name: str):
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' not found")
        return len(self.actions[action_name]['expanded_steps'])

    def get_action_steps(self, action_name: str, expanded: bool = False):
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' not found")
        if expanded:
            return self.actions[action_name]['expanded_steps']
        return self.actions[action_name]['steps']

    def get_step(self, action_name, step_idx):
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' not found")
        if step_idx >= len(self.actions[action_name]['expanded_steps']):
            raise ValueError(f"Action '{action_name}' has no step {step_idx}")
        return self.actions[action_name]['expanded_steps'][step_idx]

    def get_action_params(self, action_name: str):
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' not found")
        return self.actions[action_name]['params']

    def call(self, action_name: str, action_idx: int, params):
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' not found")

        action = self.actions[action_name]
        expanded_steps = action['expanded_steps']

        if action_idx >= len(expanded_steps):
            raise ValueError(f"Action '{action_name}' has no step {action_idx}")

        step = expanded_steps[action_idx]

        if not self.tool_registry.assert_tool_exists(step['name']):
            raise ValueError(f"Tool '{step['name']}' not found")

        result = self.tool_registry.call(step['name'], **params)

        # for data in return_data:
        #     kwargs[data] = result

        return result

    # NOTE: See ToolRegistry.acall comment above.
    # async def acall(self, action_name: str, action_idx: int, params):
    #     """Async variant of :meth:`call`.
    #
    #     Delegates to :meth:`ToolRegistry.acall` so tools can be true async
    #     (e.g. httpx).
    #     """
    #     if action_name not in self.actions:
    #         raise ValueError(f"Action '{action_name}' not found")
    #
    #     action = self.actions[action_name]
    #     expanded_steps = action['expanded_steps']
    #
    #     if action_idx >= len(expanded_steps):
    #         raise ValueError(f"Action '{action_name}' has no step {action_idx}")
    #
    #     step = expanded_steps[action_idx]
    #
    #     if not self.tool_registry.assert_tool_exists(step['name']):
    #         raise ValueError(f"Tool '{step['name']}' not found")
    #
    #     return await self.tool_registry.acall(step['name'], **params)


class ToolFactory:
    """
    Factory class for creating and initializing tools based on settings.
    """

    # Map of tool names to their corresponding functions
    # External transit API HTTP tools are async-first (httpx.AsyncClient) and are
    # not toggleable. Sync call-sites can still use ToolRegistry.call(), which
    # will bridge awaitables via run_coro_sync (and will error if misused inside
    # a running event loop).
    TOOL_MAPPING = {
        'send_get_request': send_get_request_async,
        'send_post_request': send_post_request_async,
        'format_custom_response': format_custom_response,
        'get_user_id': get_user_id,
        'query_user': query_user,
        'is_date_in_range': is_date_in_range,
        'get_selection_from_user': get_selection_from_user,
        'select_option': select_option,
        "echo_value": echo_value,
        "reduce": reduce
    }

    ENDPOINT_TOOL_MAPPING = {
        'send_request': send_request_async,
    }
    MOCK_TOOL_MAPPING = {
        'send_get_request': send_get_request_stub,
        'send_post_request': send_post_request_stub,
        'format_custom_response': format_custom_response_stub,
        'get_user_id': get_user_id_stub,
        'query_user': query_user_stub,
        'is_date_in_range': is_date_in_range_stub,
        'get_selection_from_user_stub': get_selection_from_user_stub
    }
    MOCK_ENDPOINT_TOOL_MAPPING = {
        'send_request': send_request_stub,
    }

    @classmethod
    def create_tool_registry(cls, endpoints, external_personal_info_services, debug_active=False,
                             mock_tools=0) -> ToolRegistry:
        """
        Create a ToolRegistry instance based on the given parameters.
        :param endpoints: The endpoints to be used by the registry.
        :param debug_active: Whether the debug mode is active. Defaults to False.
        :param mock_tools: The mock tools switch. Defaults to 0.
        :param external_personal_info_services: external services details
        :return: The ToolRegistry instance.
        """
        registry = ToolRegistry()

        for name in list(cls.TOOL_MAPPING.keys()):
            if debug_active and name in cls.MOCK_TOOL_MAPPING:
                # Wrap mock tool to inject mock_tools param
                def make_mock_fn(mock_fn):
                    return lambda **kwargs: mock_fn(mock_tools, **kwargs)

                registry.register(name, make_mock_fn(cls.MOCK_TOOL_MAPPING[name]))
            else:
                registry.register(name, cls.TOOL_MAPPING[name])

        for name in list(cls.ENDPOINT_TOOL_MAPPING.keys()):
            endpoint_handler = EndpointHandler(endpoints, external_personal_info_services)
            if debug_active and name in cls.MOCK_ENDPOINT_TOOL_MAPPING:
                def make_mock_endpoint_fn(mock_fn):
                    return lambda **kwargs: mock_fn(endpoint_handler, mock_tools, **kwargs)

                registry.register(name, make_mock_endpoint_fn(cls.MOCK_ENDPOINT_TOOL_MAPPING[name]))
            else:
                def make_endpoint_fn(endpoint_fn):
                    return lambda **kwargs: endpoint_fn(endpoint_handler, **kwargs)

                registry.register(name, make_endpoint_fn(cls.ENDPOINT_TOOL_MAPPING[name]))

        return registry

    @classmethod
    def create_action_registry(cls, tool_registry: ToolRegistry,
                               actions_definition: dict) -> ActionRegistry:  # TODO: Move out of tool_registry class
        action_registry = ActionRegistry(tool_registry)
        for action_name, action_def in actions_definition.items():  # Register all actions
            action_registry.register(action_name, action_def)
        # Expand all actions into sequences of tool calls
        action_registry.expand_actions()
        return action_registry
