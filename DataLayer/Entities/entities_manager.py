import os

from Core.Embeddings.embeddings import Embedder
from DataLayer.Entities.blocks import Block
from DataLayer.Entities.question import Question
from DataLayer.Entities.topic_states import TopicState
from DataLayer.Entities.workflows import Workflow
from DataLayer.RelationalDB.in_memory_entities_store import InMemoryEntitiesStore
from DataLayer.data_module import load_dict


class EntitiesManager:
    """
    High-level manager for TopicState / Workflow / Block / Question entities.

    Uses an in-memory Pandas-based store (InMemoryEntitiesStore) instead of Neo4j.
    The system should only interact with entities via this class and not access the store directly.
    """

    def __init__(self, embedder):
        if embedder is None:
            self.embedder = Embedder()
        self.embedder = embedder
        self.store: InMemoryEntitiesStore = InMemoryEntitiesStore()
        self.root_id = None

    def init_entities_store(self, data_specs, add_embeddings, root_id):
        """
        Initialize the in-memory entities store from the given data_specs.

        :param data_specs: dict with at least keys:
                - "states": dict[id -> state dict]
                - "blocks": dict[id -> block dict]
                - "workflows": dict[id -> workflow dict]
                - "personal_questions": questions spec (path + metadata)
        :param add_embeddings: whether to compute and attach embeddings via self.embedder.
        :param root_id: id of the root TopicState (used by get_root()).
        """
        # Build many-to-many workflow_blocks from workflow specs
        workflow_blocks = []
        for wf_id, wf_data in data_specs["workflows"].items():
            blocks_list = wf_data.get("blocks", [])
            for order_idx, block_ref in enumerate(blocks_list):
                block_id = block_ref.get("id")
                if not block_id:
                    continue
                workflow_blocks.append(
                    {"workflow_id": wf_id, "block_id": block_id, "order": order_idx}
                )

        # preprocess entities
        states = self.process_entities(data_specs["states"], add_embeddings=add_embeddings)
        blocks = self.process_entities(data_specs["blocks"], add_embeddings=add_embeddings)
        workflows = self.process_entities(data_specs["workflows"], add_embeddings=add_embeddings)
        questions = self.process_entities(self._load_questions(data_specs["personal_questions"]),
                                          add_embeddings=add_embeddings)

        # Load everything into the in-memory store
        self.store.load(states=states, blocks=blocks, workflows=workflows, questions=questions,
                        workflow_blocks=workflow_blocks)
        self.root_id = root_id

    def process_entities(self, entities, add_embeddings=True):
        """
        Process the entities from the Json file
        :param entities: The list of entities
        :param add_embeddings: Whether to add embeddings
        :return: The list of entities
        """
        if add_embeddings:
            entities_text = []
            for entity in entities.values():
                if "description" in entity:
                    entities_text.append(entity["title"] + "\n" + entity["description"])
                else:
                    entities_text.append(entity["title"])
            entities_embeds = self.embedder.get_text_embeddings_batch(entities_text)
            for embedding, entity in zip(entities_embeds, entities.values()):
                entity["embedding"] = embedding
        return entities

    def _load_questions(self, questions_specs):
        """
       Load questions from CSV or JSON based on questions_specs["path"].
       """
        file_type = os.path.splitext(questions_specs["path"])[1].lower()[1:]

        questions = {}  # id -> dict
        if file_type == "csv":
            import csv

            with open(questions_specs["path"], "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question_id = row.get("id").strip()
                    text = row.get("question").strip()
                    wf_id = row.get("workflow_id").strip()
                    source = row.get("source").strip()
                    questions[question_id] = {
                        "title": text,
                        "handler_workflow_id": wf_id,
                        "source": source
                    }
        elif file_type == "json":
            data = load_dict(questions_specs["path"], encoding="utf-8")
            for row in data:
                question_id = row.get("id").strip()
                text = row.get("question").strip()
                wf_id = row.get("workflow_id").strip()
                source = row.get("source").strip()
                questions[question_id] = {
                    "title": text,
                    "handler_workflow_id": wf_id,
                    "source": source
                }
        else:
            raise ValueError(f"Unsupported Question file type: {file_type}")
        return questions

    # Internal methods
    def get_root(self) -> TopicState:
        """
        Return the root TopicState. Will raise if not found.
        :return: the root TopicState
        """
        self._ensure_store_initialized()
        states = self.get_states(state_id=self.root_id)
        return states[0]

    def get_states(self, state_id=None):
        """
        Get all states, or a specific state by id.
        :param state_id: The ID of the state
        :return: List of TopicState objects matching the ID
        """
        self._ensure_store_initialized()
        if state_id:
            row = self.store.get_state_row(state_id)
            return [TopicState.from_dict(row)] if row is not None else []
        else:
            rows = self.store.get_all_state_rows()
            return [TopicState.from_dict(r) for r in rows]

    def get_children(self, state=None, state_id=None):
        """
        Get the children of a state from the graph database using either the state object or the explicit state_id
        :param state: TopicState
        :param state_id: The ID of the state
        :return: List of TopicState objects - the children of the state
        """
        self._ensure_store_initialized()
        nodes = []
        # Determine the ID from either the state object or the explicit state_id
        if state is not None:
            state_id = state.id
        if not state_id:
            return []
        else:
            rows = self.store.get_children_state_rows(parent_id=state_id)
            return [TopicState.from_dict(r) for r in rows]

    def get_workflows(self, workflow_id=None):
        """
        Get all workflows from the graph database
        :param workflow_id: The ID of the workflow
        :return: List of Workflow objects matching the ID
        """
        self._ensure_store_initialized()
        if workflow_id:
            row = self.store.get_workflow_row(workflow_id)
            return [Workflow.from_dict(row)] if row is not None else []
        else:
            rows = self.store.get_all_workflow_rows()
            return [Workflow.from_dict(r) for r in rows]

    def get_workflows_of_state(self, state=None, state_id=None, immediate=True):
        """
        Get workflows of a state.
        If immediate=True: only workflows directly under that state.
        If immediate=False: BFS over state descendants + all their workflows.

        :param immediate: Whether to get the workflows immediately or recursively
        :param state: TopicState
        :param state_id: The ID of the state
        :return: List of Workflow objects - the workflows of the state
        """
        self._ensure_store_initialized()
        if state is not None:
            state_id = state.id
        if not state_id:
            return []

        if immediate:
            rows = self.store.get_workflows_of_state_rows(state_id=state_id)
            return [Workflow.from_dict(r) for r in rows]

        # BFS over descendants
        collected = []
        visited = set()
        queue = [state_id]

        while queue:
            curr_state_id = queue.pop(0)
            if curr_state_id in visited:
                continue
            visited.add(curr_state_id)

            # workflows of current state
            rows = self.store.get_workflows_of_state_rows(state_id=curr_state_id)
            collected.extend(Workflow.from_dict(r) for r in rows)

            # children states
            child_rows = self.store.get_children_state_rows(parent_id=curr_state_id)
            for ch in child_rows:
                queue.append(ch["id"])

        return collected

    def get_workflow_parent_state(self, workflow_id: str):
        """
        Return parent state of a workflow as a single-element list.
        """
        self._ensure_store_initialized()
        row = self.store.get_workflow_parent_state_row(workflow_id)
        if row is None:
            return []
        return [TopicState.from_dict(row)]

    def get_blocks_of_state(self, state_id: str):
        """
        Return blocks that are directly attached to the given state (if modeled via parent_id).
        """
        self._ensure_store_initialized()
        rows = self.store.get_blocks_of_state_rows(state_id=state_id)
        return [Block.from_dict(r) for r in rows]

    def get_blocks_for_workflow(self, workflow_id: str):
        """
        Return blocks used by the given workflow (via workflow_blocks).
        """
        self._ensure_store_initialized()
        rows = self.store.get_blocks_for_workflow_rows(workflow_id=workflow_id)
        return [Block.from_dict(r) for r in rows]

    def get_questions_for_workflow(self, workflow_id: str, limit=None):
        """
        Return questions linked to a given workflow, ordered by id, optionally truncated to 'limit'.
        """
        self._ensure_store_initialized()
        rows = self.store.get_questions_for_workflow_rows(workflow_id=workflow_id, limit=limit)
        return [Question.from_dict(r) for r in rows]

    def search_questions_top_k(self, query_embed, vector_index_name, k=5, to_object=Question.from_dict):
        """
        Vector search over questions.


        :param query_embed: embedding vector for the query text.
        :param vector_index_name: kept for signature compatibility; ignored in this implementation.
        :param k: top-k results to return.
        :param to_object: adapter to convert row dict into a Question (defaults to Question.from_dict).

        :return: List of {"question": Question, "score": float}.
        """
        self._ensure_store_initialized()

        retrieved_quests = self.store.search_questions_top_k(query_embedding=query_embed, k=k)
        if not to_object:
            # Caller wants raw dicts
            return retrieved_quests
        else:
            out = []
            for item in retrieved_quests:
                quest = item["question"]
                q_obj = to_object(quest)
                out.append({"question": q_obj, "score": item["score"]})
            return out

    def _ensure_store_initialized(self) -> None:
        if self.store is None or self.root_id is None:
            # root_id being None usually means init_entities_store wasn't called
            raise RuntimeError("Entities store is not initialized. Call init_entities_store() / init_graph_db() first.")
