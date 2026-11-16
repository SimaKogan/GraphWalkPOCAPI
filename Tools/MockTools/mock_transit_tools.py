from Tools.MockTools.scenario_loader import load_api_response


def send_request_stub(endpoint_handler, mock_tools: int, **kwargs):
    if mock_tools in [0, 4]:
        from Tools.transit_tools import send_request
        endpoint = kwargs["endpoint"]
        add_kwargs = kwargs.get("add_kwargs", {})
        return send_request(endpoint_handler, endpoint=endpoint, add_kwargs=add_kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for send_request")
    elif mock_tools == 2:
        # endpoint_data = endpoint_handler.get_endpoint(kwargs["endpoint"])
        raise NotImplementedError("Testing has been moved to pytest suite")
    elif mock_tools == 3:
        # Scenario-driven mocks
        endpoint = kwargs["endpoint"]
        add_kwargs = kwargs.get("add_kwargs", {})
        return load_api_response(endpoint, add_kwargs)
    else:
        raise NotImplementedError("Unknown mock_tools behavior in send_request_stub")


def send_get_request_stub(mock_tools: int, **kwargs):
    if mock_tools == 0:
        from Tools.transit_tools import send_get_request
        return send_get_request(**kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for send_get_request")
    elif mock_tools == 2:
        return "Mock response"
    else:
        raise NotImplementedError("Unknown mock_tools behavior in send_get_request_stub")


def send_post_request_stub(mock_tools: int, **kwargs):
    if mock_tools == 0:
        from Tools.transit_tools import send_post_request
        return send_post_request(**kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for send_post_request")
    elif mock_tools == 2:
        return "Mock response"
    else:
        raise NotImplementedError("Unknown mock_tools behavior in send_post_request_stub")


def format_custom_response_stub(mock_tools: int, **kwargs):
    if mock_tools == 0:
        from Core.Utils.llm_utils import format_custom_response
        return format_custom_response(**kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for format_custom_response")
    elif mock_tools == 2:
        from Core.Utils.llm_utils import format_custom_response
        return format_custom_response(**kwargs)
    else:
        raise NotImplementedError("Unknown mock_tools behavior in format_custom_response_stub")


def is_date_in_range_stub(mock_tools: int, **kwargs):
    if mock_tools == 0:
        from Tools.transit_tools import is_date_in_range
        return is_date_in_range(**kwargs)
    elif mock_tools == 1:
        raise RuntimeError("Mock error for is_date_in_range")
    elif mock_tools == 2:
        return True
    elif mock_tools in [3, 4]:
        from Tools.transit_tools import is_date_in_range
        return is_date_in_range(**kwargs)
    else:
        raise NotImplementedError("Unknown mock_tools behavior in is_date_in_range_stub")
