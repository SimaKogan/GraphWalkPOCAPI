import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

from DataLayer.Entities.entity import Entity


@dataclass
class Block(Entity):
    """
    Represents an action block in a knowledge graph system
    """
    action: List[Dict[str, Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Block({super().__repr__()}, "
            f"{json.dumps(self.action)})")

    # Factory Methods
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]):
        """
        Create a Block instance from a dictionary.
        """
        return cls(
            id=state_dict["id"],
            title=state_dict.get("title", ""),
            description=state_dict.get("description", ""),
            action=json.loads(state_dict.get("action", "[]")) if isinstance(
                state_dict.get("action"), str) else state_dict.get("action", []),
            embedding=state_dict.get("embedding", [])
        )

    def get_action(self, as_json=False):
        return self.action if not as_json else self._field_as_json("action")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Block to a dictionary.
        :return: Dictionary containing all properties of the Block
        """
        data_dict = {
            **super().to_dict(),
            "action": self.get_action(as_json=True),
        }
        return data_dict