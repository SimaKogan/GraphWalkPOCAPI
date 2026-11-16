from Visualization.Graph.graph_visualization import plot_hierarchy_graph_plotly_by_level_and_terminal
from Visualization.Graph.nx_utils import json_to_graph


def visualize_graph(json_path):
    graph = json_to_graph(json_path)
    plot_hierarchy_graph_plotly_by_level_and_terminal(graph)

if __name__ == "__main__":
    visualize_graph("Resources/Specs/transit_states.json")
