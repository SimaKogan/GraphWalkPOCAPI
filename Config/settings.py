from time import sleep
from typing import Dict, Any

from pydantic_settings import BaseSettings
#
# # Pydantic automatically:
# # 1) Reads the .env file.
# # 2) Loads environment variables into the Settings class.
# # 3) Validates the types and required fields.
# #
# # Variable Priority Order (first match wins):
# # 1) Environment variables from the system
# # 2) Variables from the .env file
# # 3) Default values in the class
#
#
# class VertexSettings(BaseSettings):
#     # project_id: str
#     location: str = "us-central1"
#     model_name: str = "gemini-2.0-flash"
#     temperature: float = 0.1
#     # top_p: float = 0.95
#     # max_output_tokens: int = 1024
#
# class Neo4jSettings(BaseSettings):
#     uri: str
#     user: str
#     password: str
#     vector_index_name: str = "chunk_embedding_idx"
#
# class AppSettings(BaseSettings):
#     # max_ctx_tokens: int = 6000
#     # use_faq_shortcuts: bool = True
#     debug: bool = False
#
#
# class Settings(BaseSettings):
#     vertex: VertexSettings = VertexSettings()
#     neo4j: Neo4jSettings = Neo4jSettings()
#     CLIApp: AppSettings = AppSettings()
import json
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# load_dotenv("../.env")

# load dotenv from the project root regardless if I reach this code from main or CLIApp
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(project_root, ".env"))


def _env_first(*names: str, default: str):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass
class Settings:
    # neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    # neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    # neo4j_pass: str = os.getenv("NEO4J_PASSWORD")
    session_secret: str = os.getenv("SESSION_SECRET", "dev-secret-change-me")

    # Server/runtime tuning
    # Threadpool sizing should be an operational knob (env-driven), not stored in SystemConfigs.
    # Used by Web/threadpool_config.py.
    server_threadpool_workers: int = int(os.getenv("SERVER_THREADPOOL_WORKERS", "32"))

    # google_client_id: str = os.getenv("GOOGLE_CLIENT_ID")
    # google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET")
    google_cloud_project: str = os.getenv("GOOGLE_CLOUD_PROJECT")
    # Bearer token for the transit API. The local mock API (Tools/MockApi) accepts the default.
    transit_api_token: str = os.getenv("TRANSIT_API_TOKEN", "dev-token")

    # embedding_model: str = os.getenv("EMBED_MODEL", "text-multilingual-embedding-002")
    personal_base_url: str = _env_first(
        "PERSONAL_BASE_URL",
        "PERSONAL_SERVICE_URL",
        default="http://personal-service:8000",
    )

    general_base_url: str = _env_first(
        "GENERAL_BASE_URL",
        "GENERAL_SERVICE_URL",
        default="http://general-service:8000",
    )

    def _update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            settings_dict = dict(**json.load(f))

        self._update(**settings_dict)


settings = Settings()

# #
# class Settings(BaseSettings):
#     NEO4J_URI: str = "bolt://localhost:7687"
#     NEO4J_USER: str = "neo4j"
#     NEO4J_PASSWORD: str
#     # neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
#     # neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
#     # neo4j_pass: str = os.getenv("NEO4J_PASSWORD")
#     use_mock_tools: bool = os.getenv("USE_MOCK_TOOLS", "1") == "1"
#     # embedding_model: str = os.getenv("EMBED_MODEL", "text-multilingual-embedding-002")
#     run_config: Dict[str, Any] = {}
#
#     def update(self, path: str):
#         with open(path, "r", encoding="utf-8") as f:
#             settings_dict = dict(**json.load(f))
#
#             known_keys = self.model_fields.keys()
#             known = {k: v for k, v in settings_dict.items() if k in known_keys}
#             extras = {k: v for k, v in settings_dict.items() if k not in known_keys}
#
#             return self.model_copy(update={**known, "run_config": extras})
#
#     class Config:
#         env_file = ".env"
#         env_file_encoding = 'utf-8'
#         case_sensitive = True
#
#
# # load_dotenv()
# settings = Settings()
