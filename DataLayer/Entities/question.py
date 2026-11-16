import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

from DataLayer.Entities.entity import Entity


@dataclass
class Question(Entity):
    """
    Represents a curated question mapped to exactly one workflow in a knowledge graph system
    """
    handler_workflow_id: str = ""
    source: str = ""

    def __repr__(self) -> str:
        return (
            f"Question({super().__repr__()}, "
            f"Handler Workflow ID: {self.handler_workflow_id}, Source: {self.source}")

    # Factory Methods
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]):
        """
        Create a Question instance from a dictionary.
        """
        return cls(
            id=state_dict["id"],
            title=state_dict.get("title", ""),
            description=state_dict.get("description", ""),
            handler_workflow_id=state_dict.get("handler_workflow_id", ""),
            source=state_dict.get("source", ""),
            embedding=state_dict.get("embedding", [])
        )

    def get_handler_workflow_id(self, as_json=False):
        return self.handler_workflow_id if not as_json else self._field_as_json("handler_workflow_id")

    def get_source(self, as_json=False):
        return self.source if not as_json else self._field_as_json("source")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Question to a dictionary.
        :return: Dictionary containing all properties of the Question
        """
        data_dict = {
            **super().to_dict(),
            "handler_workflow_id": self.get_handler_workflow_id(),
            "source": self.get_source()
        }
        return data_dict

    def format_string(self) -> str:
        """
        Create a string representation of the state.
        Includes title, description,  and data elements with their names and descriptions.
        """
        return self.get_title()
