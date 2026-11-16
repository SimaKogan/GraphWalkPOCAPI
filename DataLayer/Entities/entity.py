import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np


@dataclass
class Entity:
    """
    Represents an entity in a knowledge graph system
    """
    id: str
    title: str = ""
    description: str = ""
    embedding: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Entity({self.id!r}, {self.title!r}, {self.description!r}")

    # Factory Methods
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]):
        """
        Create a Entity instance from a dictionary.
        """
        return cls(
            id=state_dict["id"],
            title=state_dict.get("title", ""),
            description=state_dict.get("description", ""),
            embedding=state_dict.get("embedding", [])
        )

    @classmethod
    def from_json_str(cls, json_str: str):
        """
        Create an Entity instance from a JSON string.
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, file_path: str):
        """
        Create an Entity instance from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # Methods to convert to JSON representations
    def _field_as_json(self, field_name: str):
        if getattr(self, field_name) is None:
            return json.dumps([])
        return json.dumps(getattr(self, field_name), ensure_ascii=False)

    def get_id(self):
        return self.id

    def get_title(self):
        return self.title

    def get_description(self):
        return self.description

    def get_embedding(self, as_np=False):
        if not as_np:
            return self.embedding
        else:
            return np.array(self.embedding)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Entity to a dictionary.
        :return: Dictionary containing all properties of the Entity
        """
        return {
            "id": self.get_id(),
            "title": self.get_title(),
            "description": self.get_description(),
            "embedding": self.get_embedding()
        }

    def format_string(self) -> str:
        """
        Create a string representation of the state.
        Includes title, description, and data elements with their names and descriptions.
        """

        parts = [f"Title: {self.title}"]
        if self.description:
            parts.append(f"Description: {self.description}")

        return "\n".join(parts)
