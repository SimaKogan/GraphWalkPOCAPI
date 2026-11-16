import time

from neo4j import GraphDatabase


class Neo4jGraphManager:
    def __init__(self, uri="neo4j://127.0.0.1:7687", database="transit", user="neo4j",
                 password=None, vector_index_name=None, create=False):
        print("Neo4jGraphManager")
        self.database = database
        self.vector_index_name = vector_index_name
        self.embedding_dim = None

        # Connect to system database to manage databases
        sys_driver = GraphDatabase.driver(uri, auth=(user, password), database="system")
        with sys_driver.session() as session:
            existing_dbs = session.run("SHOW DATABASES").data()
            db_names = [db["name"] for db in existing_dbs]

            if create:
                if self.database not in db_names:
                    session.run(f"CREATE DATABASE `{self.database}`")
                else:
                    raise ValueError(f"Database '{self.database}' already exists. ")
                    # raise ValueError(f"Database '{self.database}' does not exist and creation not allowed.")
            else:
                if self.database not in db_names:
                    raise ValueError(f"Database '{self.database}' does not exist. ")
        # Connect to the target database
        time.sleep(5)
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=self.database)

    def add_node(self, node_type: str, properties: dict, primary_key: str = None):
        """
        Create a node with the given label and properties.
        If primary_key is provided and exists in properties, merge on the primary key for uniqueness.
        Otherwise, create or merge on the full properties map (which may create duplicates if called with different data).

        :param node_type: the label\type of the node (e.g. "Chunk")
        :param properties: the properties of the node as a dict
        :param primary_key: the primary key of the node (e.g. "chunk_id")
        """
        # Open a new session (auto-closed when with-block exits)
        with self.driver.session() as session:
            if primary_key and primary_key in properties:
                # MERGE on the primary key to ensure uniqueness
                unique_value = properties[primary_key]
                # merge by primary key, set all properties on create
                query = (
                    f"MERGE (n:{node_type} {{{primary_key}: $unique}}) "
                    f"ON CREATE SET n += $props"
                )
                # Execute
                result = session.run(query, unique=unique_value, props=properties)
            else:
                # No primary key provided – merge on all properties (may create duplicates if properties differ)
                query = f"MERGE (n:{node_type} $props)"
                result = session.run(query, props=properties)

    def add_edge(self, start_node_type: str, start_match_props: dict, relationship_type: str, end_node_type: str,
                 end_match_props: dict, rel_properties: dict = None,
                 direction_forward: bool = True):
        """
        Create a relationship of type `relationship_type` between two nodes identified by their labels and properties.
        If `rel_properties` is provided, the relationship will be created with those properties.

        :param start_node_type:, end_node_type: Labels of the two nodes (e.g., 'Chunk', 'Document').
        :param start_match_props, end_match_props: Dictionaries of properties to match the start and end nodes (e.g., {'id': 'chunk1'}).
        :param relationship_type: Type of the relationship (e.g., 'PART_OF', 'NEXT').
        :param rel_properties: Optional dict of properties for the relationship.
        :param direction_forward: "forward" (default) to create (start)->(end), or "backward" to create (start)<-(end).
        """
        # MATCH patterns for the two nodes
        start_props_str = ", ".join([f"{key}: ${key}1" for key in start_match_props])
        end_props_str = ", ".join([f"{key}: ${key}2" for key in end_match_props])

        match_clause = (
            f"MATCH (a:{start_node_type} {{{start_props_str}}}) "
            f"MATCH (b:{end_node_type} {{{end_props_str}}}) "
        )
        # match_clause = (
        #         f"MATCH (a:{start_node_type} {{" + ", ".join(
        #     f"{key}: ${key}1" for key in start_match_props.keys()) + "}}) "
        #                                                              f"MATCH (b:{end_node_type} {{" + ", ".join(
        #     f"{k}: ${k}2" for k in end_match_props.keys()) + "}}) "
        # )
        # relationship direction
        if direction_forward:  # forward
            rel_pattern = f"(a)-[r:{relationship_type}]->(b)"
        else:  # "backward" direction: create relationship from b to a
            rel_pattern = f"(a)<-[r:{relationship_type}]-(b)"

        # MERGE clause for the relationship. ON CREATE to avoid duplicates.
        if rel_properties:
            prop_assignments = ", ".join([f"r.{key} = ${key}R" for key in rel_properties.keys()])
            merge_clause = f"MERGE {rel_pattern} ON CREATE SET {prop_assignments}"
        else:
            merge_clause = f"MERGE {rel_pattern}"

        # query
        cypher_query = match_clause + merge_clause

        # Combine all parameters for node matching and relationship properties into one dict
        params = {}
        # Add start node parameters (append "1" to distinguish if overlapping keys)
        for key, value in start_match_props.items():
            params[f"{key}1"] = value
        # Add end node parameters (append "2")
        for key, value in end_match_props.items():
            params[f"{key}2"] = value
        # Add relationship properties parameters (append "R")
        if rel_properties:
            for key, value in rel_properties.items():
                params[f"{key}R"] = value

        # Execute the query within a write transaction
        with self.driver.session() as session:
            session.run(cypher_query, **params)

    ##################################################

    def create_unique_constraint(self, name="unique_chunk", node_type="Chunk", key="id"):  # Primary key
        """
        Create a unique constraint on a node property.

        :param name: The name of the constraint.
        :param node_type: The label of the node to apply the constraint to.
        :param key: The property to apply the constraint to.
        """
        query = f"""
                CREATE CONSTRAINT {name} IF NOT EXISTS
                FOR (n:{node_type})
                REQUIRE n.{key} IS UNIQUE
            """
        with self.driver.session() as session:
            session.run(query)

    def get_indexes(self):
        """
        Get all indexes in the graph
        :return: List of indexes
        """
        with self.driver.session() as session:
            result = session.run(
                "SHOW INDEXES"
            )
            return [dict(record) for record in result]  # list(result)

    def create_vector_index(self, embedding_dim=768, vector_index_name="embedding_index", node_type="Chunk",
                            embedding_field="embedding"):
        """
        Create a vector index on a node property.
        :param embedding_dim: The dimension of the embeddings.
        :param vector_index_name: The name of the index.
        :param node_type: The label of the node to apply the index to.
        :param embedding_field: The property to apply the index to.
        """
        self.embedding_dim = embedding_dim
        if self.vector_index_name is None:
            self.vector_index_name = vector_index_name
        query = f"""
                CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
                FOR (c:{node_type}) ON (c.{embedding_field})
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: "cosine"
                  }}
                }};
            """

        with self.driver.session() as session:
            session.run(query)

    def search_similar_top_k(self, query_embedding, k=5, vector_index_name=None,
                             neighbors=False):
        """
        Retrieve the top-k most similar chunks to the given query embedding.
        :param query_embedding: The query embedding.
        :param k: The number of top similar chunks to retrieve.
        :param vector_index_name: The name of the vector index to use.
        :param neighbors: Whether to retrieve the neighbors of the top-k chunks.
        :return: A list of top-k chunks, each with properties and neighbors if neighbors=True.
        """

        def _get_top_k_chunks(tx, query_embed, k_top, vector_idx_name):
            result = tx.run(
                "CALL db.index.vector.queryNodes($vector_index_name, $k, $query_embed) "
                "YIELD node AS retrieved, score "
                "RETURN retrieved, score "
                "ORDER BY score DESC",
                k=k_top, query_embed=query_embed, vector_index_name=vector_idx_name
            )
            return result.data()  # list of records as dicts

        def _get_top_k_with_neighbors(tx, query_embed, k_top, vector_idx_name):
            query = """
            CALL db.index.vector.queryNodes($vector_index_name, $k, $query_embed)
            YIELD node AS retrieved, score
            OPTIONAL MATCH (retrieved)-[:NEXT]->(nextChunk)
            OPTIONAL MATCH (retrieved)<-[:NEXT]-(prevChunk)
            RETURN retrieved, score, prevChunk, nextChunk"""
            result = tx.run(query, k=k_top, query_embed=query_embed, vector_index_name=vector_idx_name)
            return result.data()

        if not vector_index_name:
            if not self.vector_index_name:
                raise ValueError("vector_index_name is not set")
            vector_index_name = self.vector_index_name

        with self.driver.session() as session:
            if neighbors:
                top_chunks = session.execute_read(_get_top_k_with_neighbors, query_embedding, k, vector_index_name)
            else:
                top_chunks = session.execute_read(_get_top_k_chunks, query_embedding, k, vector_index_name)
            return list(top_chunks)

    def get_nodes(self, node_type: str, match_props: dict, to_object=lambda r: r, order_by=None):
        """
        Query nodes in a Neo4j graph based on properties.

        :param node_type: Label of the node (e.g., 'State').
        :param match_props: Dictionary of properties to match the node (e.g., {'id': 'בלוק 1'}).
        :param to_object: Function to convert Neo4j record to domain object.
        :param order_by: Optional Cypher ORDER BY clause (e.g., 'n.level, n.title').

        :return: List of converted nodes.
        """
        if match_props is None or len(match_props) == 0:
            match_clause = f"MATCH (n:{node_type})"
            match_props = {}
        else:
            match_props_str = ", ".join([f"{key}: ${key}" for key in match_props])
            match_clause = f"MATCH (n:{node_type} {{{match_props_str}}})"

        # Final query
        cypher_query = f"""
                {match_clause}
                RETURN n {f'ORDER BY {order_by}' if order_by else ''}
                """
        with self.driver.session() as session:
            result = session.run(cypher_query, **match_props)
            return [to_object(r["n"]) for r in result]

    def get_related_nodes(self, node_type: str, match_props: dict, relationship_type: str, related_node_type: str,
                          direction_forward: bool = True, to_object=lambda r: r,
                          order_by=None):
        """
        Query related nodes in a Neo4j graph based on relationship direction and properties.

        :param node_type: Label of the starting node (e.g., 'State').
        :param match_props: Dictionary of properties to match the starting node (e.g., {'id': 'בלוק 1'}).
        :param relationship_type: Type of relationship to traverse (e.g., 'SUBSTATE_OF').
        :param related_node_type: Label of the related node to return.
        :param direction_forward: True for outgoing (start)->(related), False for incoming (start)<-(related).
        :param to_object: Function to convert Neo4j record to domain object.
        :param order_by: Optional Cypher ORDER BY clause (e.g., 'n.level, n.title').

        :return: List of converted related nodes.
        """

        match_props_str = ", ".join([f"{key}: ${key}" for key in match_props])
        match_clause = f"MATCH (a:{node_type} {{{match_props_str}}})"

        # Relationship pattern
        if direction_forward:
            rel_pattern = f"(a)-[:{relationship_type}]->(n:{related_node_type})"
        else:
            rel_pattern = f"(a)<-[:{relationship_type}]-(n:{related_node_type})"

        # Final query
        cypher_query = f"""
            {match_clause}
            MATCH {rel_pattern}
            RETURN n {f'ORDER BY {order_by}' if order_by else ''}
            """

        # Execute query
        with self.driver.session() as session:
            records = session.run(cypher_query, **match_props)
            records = records.data()
            return [to_object(record["n"]) for record in records]

    def close(self):
        self.driver.close()

    def __del__(self):
        self.close()
