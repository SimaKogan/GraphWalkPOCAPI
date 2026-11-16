import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union
import httpx

from Utils.async_utils import run_coro_sync


# NOTE:
# We intentionally do NOT keep a process-wide singleton httpx.AsyncClient here.
# In this codebase, async tools may run across multiple event loops and threads
# (e.g., via to_thread(...) + asyncio.run(...)). httpx.AsyncClient is not designed
# to be safely shared across loops/threads, so we keep per-call clients for correctness.


@dataclass
class Endpoint:
    method: str
    end_point: str
    json_request_body: list = None
    params_in_url: list = None
    data_in_url: list = None
    notes: list = None

    def __post_init__(self):
        if self.json_request_body is None:
            self.json_request_body = []
        if self.params_in_url is None:
            self.params_in_url = []
        if self.data_in_url is None:
            self.data_in_url = []
        if self.notes is None:
            self.notes = []


class EndpointHandler:
    """Resolves endpoint names (transit_endpoints.json) to HTTP calls against the transit API.

    Authentication is a plain bearer token read from the TRANSIT_API_TOKEN environment variable
    (see Config/settings.py). For local development the mock API in Tools/MockApi accepts the
    default development token.
    """

    def __init__(self, endpoints, external_personal_info_services):
        self.endpoints = {}
        self.base_url = external_personal_info_services["base_url"]
        self.external_personal_info_services = external_personal_info_services
        self._load_endpoints(endpoints)

    def _load_endpoints(self, endpoints):
        for name, endpoint_data in endpoints.items():
            self.endpoints[name] = Endpoint(**endpoint_data)

    def get_endpoint(self, endpoint_name: str):
        """Get an endpoint by name"""
        return self.endpoints.get(endpoint_name)

    def get_api_token(self) -> str:
        from Config.settings import settings
        return getattr(settings, "transit_api_token", None) or "dev-token"


def send_request(endpoint_handler, endpoint, headers=None, add_kwargs=None):
    """Synchronous wrapper for calls to the transit API.

    Requests are executed via async httpx (see :func:`send_request_async`). This sync wrapper
    exists only for sync call-sites and uses :func:`run_coro_sync` for bridging.
    """
    return run_coro_sync(send_request_async(endpoint_handler, endpoint, headers=headers, add_kwargs=add_kwargs))


async def send_request_async(endpoint_handler, endpoint, headers=None, add_kwargs=None):
    """Async variant of send_request.

    Used when running workflows inside async FastAPI handlers.
    """
    endpoint_data = endpoint_handler.get_endpoint(endpoint)
    if endpoint_data is None:
        raise KeyError(f"Unknown endpoint: {endpoint}")
    headers = dict(headers or {})
    headers["Authorization"] = f"Bearer {endpoint_handler.get_api_token()}"

    url = endpoint_handler.base_url + endpoint_data.end_point

    if endpoint_data.method == "POST":
        json_body = {param_key: (add_kwargs or {}).get(param_key, None) for param_key in
                     endpoint_data.json_request_body}
        data_in_url = {param_key: (add_kwargs or {}).get(param_key, None) for param_key in endpoint_data.data_in_url}
        response = await send_post_request_async(url, json=json_body, data=data_in_url, headers=headers)
    elif endpoint_data.method == "GET":
        params = {param_key: (add_kwargs or {}).get(param_key, None) for param_key in endpoint_data.params_in_url}
        response = await send_get_request_async(url, params=params, headers=headers)
    else:
        raise Exception(f"Unsupported method: {endpoint_data.method}")

    response = trim_response(response['data'][0])
    return response


def send_get_request(url, params=None, headers=None):
    """Sync wrapper delegating to :func:`send_get_request_async`."""
    return run_coro_sync(send_get_request_async(url, params=params, headers=headers))


async def send_get_request_async(url, params=None, headers=None):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise Exception(f"GET request failed: {e}")


def send_post_request(url, data=None, json=None, headers=None):
    """Sync wrapper delegating to :func:`send_post_request_async`."""
    return run_coro_sync(send_post_request_async(url, data=data, json=json, headers=headers))


async def send_post_request_async(url, data=None, json=None, headers=None):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, data=data, json=json, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise Exception(f"POST request failed: {e}")


def unify_value_format(value):
    if isinstance(value, str) and is_iso8601_date(value):
        return parse_iso_8601(value)
    else:
        return value


def echo_value(value):
    """
    Simple helper that returns its input as-is.
    Used to persist action parameters into facts via return_data.
    """
    return value


def reduce(val_a, val_b, operator):
    if operator == "add":
        return val_a + val_b
    elif operator == "subtract":
        return val_a - val_b
    elif operator == "multiply":
        return val_a * val_b
    elif operator == "divide":
        return val_a / val_b
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def check_response_detail_validity(details):
    """
    Check if a response field is valid.

    A field of an API response looks like::

        {"fieldNameEn": "...", "fieldNameHe": "...", "value": ..., "remark": null, "responseType": 1}

    :param details: a single response field (or a list of rows)
    :return: True if the field has both display names, False otherwise
    """
    if not details:
        return False
    if isinstance(details, list):
        return True
    elif 'fieldNameEn' not in details or details['fieldNameEn'] == "":
        return False
    elif 'fieldNameHe' not in details or details['fieldNameHe'] == "":
        return False
    else:
        return True


def trim_response(response):
    """Keeps only the valid fields of an API response."""
    trimmed_response = {}
    for key, details in response.items():
        if check_response_detail_validity(details):
            trimmed_response[key] = details
    return trimmed_response


DATE_LIKE_RE = re.compile(r"^\d{4}(-\d{2}-\d{2})?$")


def _to_datetime(x: Union[str, int, datetime, None]):
    if x is None:
        return None

    if isinstance(x, datetime):
        return x

    if isinstance(x, int):
        if 0 <= x <= 9999:
            return datetime(x, 1, 1)
        return None

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None

        sl = s.lower()
        if sl in {"now", "today"}:
            return datetime.now()

        if len(s) == 4 and s.isdigit():
            return datetime(int(s), 1, 1)

        if DATE_LIKE_RE.fullmatch(s):
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None

        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    return None


def is_date_in_range(start_date: Union[str, int, datetime], date_to_check: Union[str, int, datetime],
                     end_date: Optional[Union[str, int, datetime]] = None, year_only: bool = False) -> bool:
    """
    Checks if a date is in the range of two dates using date-based comparison.

    Special handling of string inputs:
    - An empty string ("") is treated as unspecified input and returns True.
    - Any non-empty string that is not in YYYY or YYYY-MM-DD format returns False.
    - Invalid date values return False.

    If end_date is None, checks if the date_to_check is after the start_date.

    :param start_date: Start date (datetime, YYYY-MM-DD string, YYYY string, or YYYY int)
    :param date_to_check: Date to check (empty string returns True; invalid date returns False)
    :param end_date: End date (datetime, YYYY-MM-DD string, YYYY string, or YYYY int)
    :param year_only: If True, checks if the year of the date_to_check is in the range

    :return: bool: True if the date is in the range or if date_to_check is empty; False otherwise
    """

    # 🔹 Handle strings explicitly
    if isinstance(date_to_check, str):
        s = date_to_check.strip()
        if s == "":
            return True
        if not DATE_LIKE_RE.fullmatch(s):
            return False

    start_dt = _to_datetime(start_date)
    check_dt = _to_datetime(date_to_check)
    end_dt = _to_datetime(end_date) if end_date is not None else None

    # Parsing failure => False
    if start_dt is None or check_dt is None or (end_date is not None and end_dt is None):
        return False

    if year_only:
        sy, cy = start_dt.year, check_dt.year
        if end_dt:
            return sy <= cy <= end_dt.year
        return sy <= cy

    if end_dt:
        return start_dt <= check_dt <= end_dt
    return start_dt <= check_dt


def create_date_iso_8601(date_str="01-06-2025 08:05:07", date_str_format="%d-%m-%Y %H:%M:%S"):
    dt = datetime.strptime(date_str, date_str_format)
    dt = dt.replace(tzinfo=timezone.utc)  # TODO: change to local timezone?
    return dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def is_iso8601_date(s):
    # Regex pattern for ISO 8601 with optional milliseconds and timezone
    iso8601_pattern = re.compile(
        r'^(\d{4}-\d{2}-\d{2})'  # Date part YYYY-MM-DD
        r'T(\d{2}:\d{2}:\d{2})'  # Time part HH:MM:SS
        r'(\.\d{1,6})?'  # Optional fractional seconds
        r'(Z|[+-]\d{2}:\d{2})?$'  # UTC Z or timezone offset
    )
    return bool(iso8601_pattern.match(s))


def parse_iso_8601(iso_str="2022-06-01T08:05:07.467Z", str_format="%d-%m-%Y"):
    # Replace 'Z' with '+00:00' to make it compatible with fromisoformat
    iso_str = iso_str.replace('Z', '+00:00')
    dt = datetime.fromisoformat(iso_str)
    if str_format:
        return dt.strftime(str_format)
    return dt
