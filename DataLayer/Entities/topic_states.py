import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

from DataLayer.Entities.entity import Entity


@dataclass
class TopicState(Entity):
    """
    Represents the state of a topic in a knowledge graph system
    """
    level: int = 0
    # exclusions: List[str] = field(default_factory=list)
    kind: str = ""

    def __repr__(self) -> str:
        return (
            f"TopicState({super().__repr__()}, "
            f"{self.level}, {self.kind})")

    # Factory Methods
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "TopicState":
        """
        Create a TopicState instance from a dictionary.
        """
        return cls(
            id=state_dict["id"],
            title=state_dict.get("title", ""),
            description=state_dict.get("description", ""),
            level=state_dict.get("level", 0),
            kind=state_dict.get("kind", ""),
            # exclusions=json.loads(state_dict.get("exclusions", "[]")) if isinstance(
            #     state_dict.get("exclusions"), str) else state_dict.get("exclusions", []),
            embedding=state_dict.get("embedding", [])
        )

    def get_level(self):
        return self.level

    def get_kind(self):
        return self.kind

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the TopicState to a dictionary.
        :return: Dictionary containing all properties of the TopicState
        """
        data_dict = {
            **super().to_dict(),
            "level": self.get_level(),
            # "exclusions": self.get_exclusions(as_json=True),
            "kind": self.get_kind(),
        }
        return data_dict

    def format_string(self) -> str:
        """
        Create a string representation of the state.
        Includes title, description, and data elements with their names and descriptions.
        """

        parts = [super().format_string()]

        return "\n".join(parts)
