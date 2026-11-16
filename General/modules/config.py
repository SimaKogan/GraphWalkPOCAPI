import os

# Database settings (read from the environment, see .env.example)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# One collection per topic of General/modules/topics_catalog.py
DB_NAMES = [
    "fares_and_tickets",
    "passes",
    "discounts",
    "refunds",
    "card",
    "fines",
    "accessibility",
    "lost_property",
    "network_and_schedules",
]

# Weaviate settings
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS = "HebrewContent"
