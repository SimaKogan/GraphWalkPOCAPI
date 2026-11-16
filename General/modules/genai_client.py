"""Shared Gemini client factory for the General service.

The Google Cloud project and region come from the environment (see .env.example):
GOOGLE_CLOUD_PROJECT and VERTEXAI_LOCATION.
"""
import os

from google import genai

DEFAULT_MODEL = os.getenv("GENERAL_MODEL", "gemini-2.5-flash")


def get_client() -> "genai.Client":
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("VERTEXAI_LOCATION", "us-central1"),
    )
