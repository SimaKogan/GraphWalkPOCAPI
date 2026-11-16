from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class InMemoryEntitiesStore:
    """
    In-memory 'relational' store for TopicState / Block / Workflow / Question
    based on Pandas DataFrames + numpy vector search.
    """

    def __init__(self) -> None:
        # Core tables
        self.states_df: pd.DataFrame = pd.DataFrame()
        self.blocks_df: pd.DataFrame = pd.DataFrame()
        self.workflows_df: pd.DataFrame = pd.DataFrame()
        self.questions_df: pd.DataFrame = pd.DataFrame()
        # Many-to-many: workflows ↔ blocks
        self.workflow_blocks_df: pd.DataFrame = pd.DataFrame()  # cols: workflow_id, block_id, order

        # Vector search cache for questions
        self._question_emb_matrix: Optional[np.ndarray] = None
        self._question_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, states: Dict[str, Dict[str, Any]], blocks: Dict[str, Dict[str, Any]],
             workflows: Dict[str, Dict[str, Any]], questions: Dict[str, Dict[str, Any]],
             workflow_blocks: List[Dict[str, Any]]):
        """
        Load all entities and relations into DataFrames.

        Args:
            states, blocks, workflows, questions:
                dicts of the form {id: {field: value, ...}}.
            workflow_blocks:
                list of dicts like {"workflow_id": ..., "block_id": ..., "order": ...}.
        """
        self.states_df = self._dict_to_df(states)
        self.blocks_df = self._dict_to_df(blocks)
        self.workflows_df = self._dict_to_df(workflows)
        self.questions_df = self._dict_to_df(questions)
        self.workflow_blocks_df = self._list_to_df(
            workflow_blocks, expected_cols=["workflow_id", "block_id", "order"]
        )

        self._build_question_embedding_matrix()

    @staticmethod
    def _dict_to_df(entities: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        if not entities:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(entities, orient="index")
        df.index.name = "id"
        df.reset_index(inplace=True)
        return df

    @staticmethod
    def _list_to_df(rows: List[Dict[str, Any]], expected_cols: List[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=expected_cols)
        df = pd.DataFrame(rows)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        return df

    def _build_question_embedding_matrix(self) -> None:
        """
        Pre-compute a dense embedding matrix for questions to enable fast cosine similarity search.
        """
        if self.questions_df.empty or "embedding" not in self.questions_df.columns:
            self._question_emb_matrix = None
            self._question_ids = None
            return

        emb_list: List[np.ndarray] = []
        ids: List[str] = []
        for _, row in self.questions_df.iterrows():
            emb = row.get("embedding")
            if emb is None:
                continue
            arr = np.asarray(emb, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            emb_list.append(arr)
            ids.append(row["id"])

        if not emb_list:
            self._question_emb_matrix = None
            self._question_ids = None
            return

        mat = np.stack(emb_list, axis=0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._question_emb_matrix = mat / norms
        self._question_ids = np.array(ids)

    # ------------------------------------------------------------------
    # States: row-level access
    # ------------------------------------------------------------------
    def get_state_row(self, state_id: str) -> Optional[Dict[str, Any]]:
        if self.states_df.empty:
            return None
        rows = self.states_df[self.states_df["id"] == state_id]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def get_all_state_rows(self) -> List[Dict[str, Any]]:
        if self.states_df.empty:
            return []
        df = self.states_df.copy()
        sort_cols = [c for c in ("level", "title") if c in df.columns]
        if sort_cols:
            df.sort_values(by=sort_cols, inplace=True)
        return df.to_dict(orient="records")

    def get_children_state_rows(self, parent_id: str) -> List[Dict[str, Any]]:
        if self.states_df.empty or "parent_id" not in self.states_df.columns:
            return []
        df = self.states_df[self.states_df["parent_id"] == parent_id].copy()
        sort_cols = [c for c in ("level", "title") if c in df.columns]
        if sort_cols:
            df.sort_values(by=sort_cols, inplace=True)
        return df.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Workflows: row-level access
    # ------------------------------------------------------------------
    def get_workflow_row(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        if self.workflows_df.empty:
            return None
        rows = self.workflows_df[self.workflows_df["id"] == workflow_id]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def get_all_workflow_rows(self) -> List[Dict[str, Any]]:
        if self.workflows_df.empty:
            return []
        df = self.workflows_df.copy()
        if "title" in df.columns:
            df.sort_values(by="title", inplace=True)
        return df.to_dict(orient="records")

    def get_workflows_of_state_rows(self, state_id: str) -> List[Dict[str, Any]]:
        if self.workflows_df.empty or "parent_id" not in self.workflows_df.columns:
            return []
        df = self.workflows_df[self.workflows_df["parent_id"] == state_id].copy()
        if "title" in df.columns:
            df.sort_values(by="title", inplace=True)
        return df.to_dict(orient="records")

    def get_workflow_parent_state_row(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the parent state's row (dict) of the given workflow, or None if not found.
        """
        wf = self.get_workflow_row(workflow_id)
        if wf is None:
            return None
        parent_id = wf.get("parent_id")
        if not parent_id:
            return None
        return self.get_state_row(parent_id)

    # ------------------------------------------------------------------
    # Blocks: row-level access
    # ------------------------------------------------------------------
    def get_blocks_of_state_rows(self, state_id: str) -> List[Dict[str, Any]]:
        """
        Blocks directly attached to a state via their parent_id (if modeled that way).
        """
        if self.blocks_df.empty or "parent_id" not in self.blocks_df.columns:
            return []
        df = self.blocks_df[self.blocks_df["parent_id"] == state_id]
        return df.to_dict(orient="records")

    def get_blocks_for_workflow_rows(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Blocks used by a workflow (many-to-many via workflow_blocks_df).
        """
        if self.workflow_blocks_df.empty or self.blocks_df.empty:
            return []
        rel = self.workflow_blocks_df[self.workflow_blocks_df["workflow_id"] == workflow_id]
        if rel.empty:
            return []
        block_ids = rel["block_id"].unique().tolist()
        df = self.blocks_df[self.blocks_df["id"].isin(block_ids)].copy()

        # Optional ordering by 'order' from workflow_blocks_df
        if "order" in self.workflow_blocks_df.columns:
            order_map = {
                (row["workflow_id"], row["block_id"]): row["order"]
                for _, row in self.workflow_blocks_df.iterrows()
            }
            df["__order"] = df["id"].map(lambda bid: order_map.get((workflow_id, bid), 0))
            df.sort_values(by="__order", inplace=True)
            df.drop(columns="__order", inplace=True)

        return df.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Questions: row-level access
    # ------------------------------------------------------------------
    def get_questions_for_workflow_rows(
            self,
            workflow_id: str,
            limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.questions_df.empty or "handler_workflow_id" not in self.questions_df.columns:
            return []
        df = self.questions_df[self.questions_df["handler_workflow_id"] == workflow_id].copy()
        if "id" in df.columns:
            df.sort_values(by="id", inplace=True)
        if limit is not None:
            df = df.head(limit)
        return df.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Vector search on questions
    # ------------------------------------------------------------------
    def search_questions_top_k(self, query_embedding: Any, k: int = 5):  # TODO: All entities
        """
        Cosine similarity search over question embeddings.

        Returns:
            List of dicts:
            [{"question": <question_row_dict>, "score": float}, ...] sorted by score desc.
        """
        if self._question_emb_matrix is None or self._question_ids is None:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1:
            q = q.reshape(-1)

        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return []
        q = q / q_norm

        mat = self._question_emb_matrix
        sims = mat @ q  # cosine similarity - matmul (rows already normalized)

        k = min(k, sims.shape[0])
        if k <= 0:
            return []

        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        # For quick lookup question_id -> row dict
        id_to_row = {row["id"]: row for row in self.questions_df.to_dict(orient="records")}

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            q_id = self._question_ids[idx]
            row = id_to_row.get(q_id)
            if row is None:
                continue
            score = float(sims[idx])
            results.append({"question": row, "score": score})
        return results
