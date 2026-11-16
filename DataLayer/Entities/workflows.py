import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

from DataLayer.Entities.entity import Entity


@dataclass
class Workflow(Entity):
    """
    Represents a workflow block in a knowledge graph system
    """
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    _adjacency: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Workflow({super().__repr__()}, "
            f"{json.dumps(self.blocks)}, {json.dumps(self.edges)})")

    # Factory Methods
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]):
        """
        Create a Workflow instance from a dictionary.
        """
        return cls(
            id=state_dict["id"],
            title=state_dict.get("title", ""),
            description=state_dict.get("description", ""),
            blocks=json.loads(state_dict.get("blocks", "[]")) if isinstance(
                state_dict.get("blocks"), str) else state_dict.get("blocks", []),
            edges=json.loads(state_dict.get("edges", "[]")) if isinstance(
                state_dict.get("edges"), str) else state_dict.get("edges", []),
            embedding=state_dict.get("embedding", []),
            metadata=json.loads(state_dict.get("metadata", "{}")) if isinstance(
                state_dict.get("metadata"), str) else state_dict.get("metadata", {})
        )

    def get_blocks(self, as_json=False):
        return self.blocks if not as_json else self._field_as_json("blocks")

    def get_edges(self, as_json=False):
        return self.edges if not as_json else self._field_as_json("edges")

    def get_metadata(self, as_json=False):
        return self.metadata if not as_json else self._field_as_json("metadata")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Block to a dictionary.
        :return: Dictionary containing all properties of the Block
        """
        data_dict = {
            **super().to_dict(),
            "blocks": self.get_blocks(as_json=True),
            "edges": self.get_edges(as_json=True),
            "metadata": self.get_metadata(as_json=True),
        }
        return data_dict

    def get_adjacency(self):
        """
        Lazily build (and cache) an adjacency map: {from_id: [edge, ...]}.
        """
        if self._adjacency:
            return self._adjacency
        adj: Dict[str, List[Dict[str, Any]]] = {}
        for e in (self.edges or []):
            adj.setdefault(e["from"], []).append(e)
        self._adjacency = adj
        return self._adjacency

    def start_queue(self):
        """
        Deterministic (serialized) start: list of node ids pointed to by 'start' edges.
        If none, returns ['end'] so the executor exits immediately.
        """
        adj = self.get_adjacency()
        return [e["to"] for e in adj.get("start", [])] or ["end"]

    def format_string(self) -> str:
        """
        Create a string representation of the workflow.
        """

        parts = ["Workflow ID: " + self.get_id(), super().format_string()]

        return "\n".join(parts)
