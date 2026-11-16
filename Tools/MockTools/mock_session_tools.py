from Tools.MockTools.scenario_loader import load_scenario, get_current_scenario_id
from Config.settings import settings


def get_user_id_stub(mock_tools: int, **kwargs):  # prompt_text: str = "מה תעודת הזהות שלך?"):
    if mock_tools == 0:
        from Tools.session_tools import get_user_id
        return get_user_id(**kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for get_user_id")
    elif mock_tools == 2:
        # from Tools.session_tools import get_user_id
        # return get_user_id(**kwargs)
        raise RuntimeError("mock_tools == 2: configure manual ID or migrate to scenario mode")
    elif mock_tools in [3, 4]:
        # scenario-driven value
        scenario = load_scenario()
        user_id = scenario.get("user_id")
        if not user_id:
            raise RuntimeError(
                f"Scenario '{get_current_scenario_id()}' must define 'user_id' for get_user_id_stub"
            )
        return str(user_id)
    else:
        raise NotImplementedError("Unknown mock_tools behavior in get_user_id")


def query_user_stub(mock_tools: int, **kwargs):  # try_to_infer_from_conv, prompt, return_keys, description, return_type
    if mock_tools == 0:
        from Tools.session_tools import query_user
        return query_user(**kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for query_user")
    elif mock_tools in [2, 3, 4]:
        # return {field: f"{field}_mock" for field in kwargs["return_keys"]}
        from Tools.session_tools import query_user
        return query_user(**kwargs)
    else:
        raise NotImplementedError("Unknown mock_tools behavior in query_user")


def get_selection_from_user_stub(mock_tools: int, **kwargs):
    if mock_tools == 0:
        from Tools.session_tools import get_selection_from_user
        return get_selection_from_user(**kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for get_selection_from_user")
    elif mock_tools in [2, 3, 4]:
        # return {field: f"{field}_mock" for field in kwargs["return_keys"]}
        from Tools.session_tools import get_selection_from_user
        return get_selection_from_user(**kwargs)
    else:
        raise NotImplementedError("Unknown mock_tools behavior in get_selection_from_user")
