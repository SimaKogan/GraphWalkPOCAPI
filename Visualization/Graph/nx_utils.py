import networkx as nx
import json


def json_to_graph(json_path):
    """
    Load a networkx Digraph from a JSON file
    :param json_path: str
    :return: nx.Graph
    """
    with open(json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    G = nx.DiGraph()

    # Add nodes
    for node_id, node in graph_data["states"].items():
        attrs = {k: v for k, v in node.items() if k != "id" and k != "parent"}
        G.add_node(node_id, **attrs)

    for node_id, node in graph_data["states"].items():
        parent = node["parent_id"]
        if parent is not None and parent != "root":
            G.add_edge(parent, node_id)
    # # Add edges
    # for edge in graph_data["edges"]:
    #     source = edge["source"]
    #     target = edge["target"]
    #     attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}
    #     G.add_edge(source, target, **attrs)

    return G


def graph_to_json(G, dest_path):
    """
    Save a graph as a JSON file
    :param G: nx.Graph
    :param dest_path: str
    """
    nodes = []
    for node, data in G.nodes(data=True):
        node_entry = {"id": node}
        node_entry.update(data)  # Includes attributes like 'type', 'synonyms'
        nodes.append(node_entry)

    edges = []
    for source, target, data in G.edges(data=True):
        edge_entry = {"source": source, "target": target}
        edge_entry.update(data)  # Includes attributes like 'label'
        edges.append(edge_entry)

    graph_json = {
        "nodes": nodes,
        "edges": edges
    }

    graph_json_str = json.dumps(graph_json, ensure_ascii=False, indent=2)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(graph_json_str)
