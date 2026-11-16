import os
import json
import subprocess

from google.auth import default
from google.auth.transport.requests import Request
import vertexai
from google import genai

LOCATION = os.getenv("VERTEXAI_LOCATION", "us-central1")


def get_gcloud_defaults():
    credentials, project = default()

    # Refresh creds
    if credentials and getattr(credentials, "expired", False) and getattr(credentials, "refresh_token", None):
        credentials.refresh(Request())
    if not project:
        project = (
                os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("GCLOUD_PROJECT")
                or os.getenv("GCP_PROJECT")
        )

    if not project:
        raise RuntimeError(
            "Could not resolve GCP project from ADC or environment. "
            "Set GOOGLE_CLOUD_PROJECT (or GCLOUD_PROJECT / GCP_PROJECT)."
        )

    return credentials, project


def init_vertexai():
    _, project = _, project = get_gcloud_defaults()

    vertexai.init(project=project, location=LOCATION)
    return project


def init_genai_client():
    _, project = get_gcloud_defaults()

    client = genai.Client(
        vertexai=True,
        project=project,
        location=LOCATION
    )
    return client
