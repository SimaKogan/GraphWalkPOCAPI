from tqdm import tqdm

from DataLayer.Entities.blocks import Block
from DataLayer.Entities.question import Question
from DataLayer.Entities.workflows import Workflow
from DataLayer.Entities.topic_states import TopicState


def add_states_to_graph_db(states, graph_manager, vector_index_name=None):
    add_entities(states, graph_manager=graph_manager, node_type="State", key="id", create_unique_constraints=True,
                 unique_constraint_name="unique_state", entity_from_dict=TopicState.from_dict)
    add_part_of_relationships(states, graph_manager,
                              start_node_type="State", start_key="id", start_match_prop="id",
                              end_node_type="State", end_key="parent_id", end_match_prop="id",
                              relationship_type="CHILD_OF")
    # add_part_of_relationships(states, graph_manager,
    #                           start_node_type="State", start_key="parent", start_match_prop="id",
    #                           end_node_type="State", end_key="id", end_match_prop="id",
    #                           relationship_type="PARENT_OF")
    graph_manager.create_vector_index(node_type="State", vector_index_name=vector_index_name,
                                      embedding_field="embedding")
    # pprint(graph_manager.get_indexes())
    return graph_manager


def add_blocks_to_graph_db(blocks, graph_manager,
                           vector_index_name=None):  # NOTE: we don't really need blocks in the graph. just for readability
    add_entities(blocks, graph_manager=graph_manager, node_type="Block", key="id", create_unique_constraints=True,
                 unique_constraint_name="unique_block", entity_from_dict=Block.from_dict)
    add_part_of_relationships(blocks, graph_manager,
                              start_node_type="Block", start_key="id", start_match_prop="id",
                              end_node_type="State", end_key="parent_id", end_match_prop="id",
                              relationship_type="BLOCK_OF")
    graph_manager.create_vector_index(node_type="Block", vector_index_name=vector_index_name,
                                      embedding_field="embedding")
    return graph_manager


def add_workflows_to_graph_db(workflows, graph_manager, vector_index_name=None):
    add_entities(workflows, graph_manager=graph_manager, node_type="Workflow", key="id", create_unique_constraints=True,
                 unique_constraint_name="unique_workflow", entity_from_dict=Workflow.from_dict)
    add_part_of_relationships(workflows, graph_manager,
                              start_node_type="Workflow", start_key="id", start_match_prop="id",
                              end_node_type="State", end_key="parent_id", end_match_prop="id",
                              relationship_type="WORKFLOW_OF")
    add_workflow_blocks(workflows, graph_manager)
    graph_manager.create_vector_index(node_type="Workflow", vector_index_name=vector_index_name,
                                      embedding_field="embedding")
    return graph_manager


def add_questions_to_graph_db(questions, graph_manager, vector_index_name=None):
    add_entities(questions, graph_manager=graph_manager, node_type="Question", key="id", create_unique_constraints=True,
                 unique_constraint_name="unique_question", entity_from_dict=Question.from_dict)
    add_part_of_relationships(questions, graph_manager,
                              start_node_type="Question", start_key="id", start_match_prop="id",
                              end_node_type="Workflow", end_key="handler_workflow_id", end_match_prop="id",
                              relationship_type="QUESTION_OF")
    graph_manager.create_vector_index(node_type="Question", vector_index_name=vector_index_name,
                                      embedding_field="embedding")
    return graph_manager


def add_workflow_blocks(workflows, graph_manager):
    for workflow_id, workflow in workflows.items():
        for block in workflow["blocks"]:
            graph_manager.add_edge(start_node_type="Workflow", start_match_props={"id": workflow_id},
                                   relationship_type="USES_BLOCK",
                                   end_node_type="Block", end_match_props={"id": block["id"]})


def add_part_of_relationships(relationships, graph_manager,
                              start_node_type="Chunk", start_key="chunk_id", start_match_prop="id",
                              end_node_type="Document", end_key="source_name", end_match_prop="source_name",
                              relationship_type="PART_OF"):
    # relationships arg is now a dict keyed by id
    for rel_id, rel_data in tqdm(relationships.items()):
        graph_manager.add_edge(
            start_node_type=start_node_type,
            start_match_props={start_match_prop: rel_id if start_key == "id" else rel_data[start_key]},
            relationship_type=relationship_type,
            end_node_type=end_node_type,
            end_match_props={end_match_prop: rel_data[end_key]}
        )


# def add_part_of_relationships(relationships, graph_manager, start_node_type="Chunk", start_key="chunk_id",
#                               start_match_prop="id", end_node_type="Document", end_key="source_name",
#                               end_match_prop="source_name", relationship_type="PART_OF"):
#     for relationship in tqdm(relationships):
#         graph_manager.add_edge(start_node_type=start_node_type,
#                                start_match_props={start_match_prop: relationship[start_key]},
#                                relationship_type=relationship_type,
#                                end_node_type=end_node_type, end_match_props={end_match_prop: relationship[end_key]})


def add_entities(entities, entity_from_dict, graph_manager, node_type="State", key="id",
                 create_unique_constraints=True, unique_constraint_name="unique_state"):
    for entity_id, entity_details in entities.items():
        entity = entity_from_dict({"id": entity_id, **entity_details})
        node = graph_manager.add_node(node_type=node_type, primary_key=key, properties=entity.to_dict())
    # Create Constraint
    if create_unique_constraints and unique_constraint_name:
        graph_manager.create_unique_constraint(name=unique_constraint_name, node_type=node_type, key=key)
