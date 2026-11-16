import plotly.express as px
from collections import defaultdict
import plotly.graph_objects as go


def plot_hierarchy_graph_plotly(graph, root_type=0):
    """
    Plot a hierarchy graph using Plotly with discrete color legend
    :param graph: nx.DiGraph
    :param root_type: str
    :return:
    """
    # Group nodes by type
    type_to_nodes = defaultdict(list)
    for node, data in graph.nodes(data=True):
        node_type = data.get('level')
        type_to_nodes[node_type].append(node)

    # Calculate average out-degree per type
    type_out_degrees = {
        t: sum(graph.out_degree(n) for n in nodes) / len(nodes)
        for t, nodes in type_to_nodes.items()
    }

    # Sort types: higher out-degree types higher up
    sorted_types = sorted(type_out_degrees.items(), key=lambda x: -x[1])  # Descending out-degree
    type_levels = {t: i for i, (t, _) in enumerate(sorted_types)}

    # Optional: ensure root_type is always at the top (level 0)
    if root_type in type_levels:
        root_level = type_levels[root_type]
        for t in type_levels:
            if type_levels[t] < root_level:
                type_levels[t] += 1
        type_levels[root_type] = 0

    # Assign colors to types
    type_colors = {t: px.colors.qualitative.Light24[i % len(px.colors.qualitative.Light24)]
                   for i, t in enumerate(type_levels)}

    # Build node coordinates and visuals
    node_x, node_y, node_color, node_size, node_text, node_labels = [], [], [], [], [], []
    for i, node in enumerate(graph.nodes()):
        node_type = graph.nodes[node]['level']
        x = i * 2  # Horizontal spread
        y = -type_levels[node_type] * 2  # Vertical by level (lower types appear lower)
        node_x.append(x)
        node_y.append(y)
        node_color.append(type_colors[node_type])
        node_size.append(10 + 5 * graph.degree(node))
        node_text.append(f"Node {node}<br>Type: {node_type}<br>Degree: {graph.degree(node)}")
        node_labels.append(str(node))

    # Build edge coordinates
    edge_x, edge_y = [], []
    node_list = list(graph.nodes())
    for src, dst in graph.edges():
        x0, y0 = node_x[node_list.index(src)], node_y[node_list.index(src)]
        x1, y1 = node_x[node_list.index(dst)], node_y[node_list.index(dst)]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Plot edges
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(color='gray', width=1), hoverinfo='none')

    # Plot nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_labels, textposition='top center',
        hoverinfo='text', hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='black')),
        showlegend=False
    )

    # Legend
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(size=10, color=color), name=f"Type {t}")
        for t, color in type_colors.items()
    ]

    # Final figure
    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces,
                    layout=go.Layout(
                        title="Hierarchical Network Graph by Out-Degree",
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    fig.show()


def plot_hierarchy_graph_plotly_by_level(graph, level_attr='level', root_level=0, level_gap=2.0, x_gap=2.0):
    """
    Plot a hierarchy graph using Plotly where vertical placement is determined by the node's `level` attribute.
    :param graph: nx.DiGraph
    :param level_attr: node attribute that stores the level (default: 'level')
    :param root_level: optional level to pin at the top (y=0)
    :param level_gap: vertical distance between adjacent levels
    :param x_gap: horizontal spacing between nodes within the same level
    """

    # Group nodes by level
    level_to_nodes = defaultdict(list)
    for node, data in graph.nodes(data=True):
        lvl = data.get(level_attr, 0)
        level_to_nodes[lvl].append(node)

    # Normalize levels so that root_level is at y=0 (optional)
    unique_levels = sorted(level_to_nodes.keys())
    if root_level in unique_levels:
        # shift so root_level -> 0, levels below it are positive (downwards)
        shift = -unique_levels.index(root_level)
        ordered_levels = [lvl for lvl in unique_levels]  # keep order
        level_index = {lvl: i + shift for i, lvl in enumerate(ordered_levels)}
    else:
        # just index in sorted order starting from 0 at the top
        level_index = {lvl: i for i, lvl in enumerate(unique_levels)}

    # Assign colors by level
    palette = px.colors.qualitative.Light24
    level_colors = {lvl: palette[i % len(palette)] for i, lvl in enumerate(sorted(unique_levels))}

    # Build node coordinates and visuals
    node_positions = {}
    node_x, node_y, node_color, node_size, node_text, node_labels = [], [], [], [], [], []

    for lvl in sorted(level_to_nodes.keys(), key=lambda x: level_index[x]):
        nodes_at_level = level_to_nodes[lvl]
        # spread nodes horizontally around 0
        n = len(nodes_at_level)
        xs = [x_gap * (i - (n - 1) / 2.0) for i in range(n)]
        y = -level_gap * level_index[lvl]

        for x, node in zip(xs, nodes_at_level):
            node_positions[node] = (x, y)
            node_x.append(x)
            node_y.append(y)
            node_color.append(level_colors[lvl])
            deg = graph.degree(node)
            node_size.append(10 + 5 * deg)  # keep degree only for size; change if you prefer fixed sizes
            # avoid backslashes inside f-string expressions (syntax error on some Python versions)
            meta_lines = [f"{k}: {v}" for k, v in graph.nodes[node].items() if v]
            meta_html = ("<br>" + "<br>".join(meta_lines)) if meta_lines else ""
            node_text.append(f"Node {node}<br>Level: {lvl}{meta_html}")
            node_labels.append(str(node))

    # Build edge coordinates
    edge_x, edge_y = [], []
    for src, dst in graph.edges():
        x0, y0 = node_positions[src]
        x1, y1 = node_positions[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Plot edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='gray', width=1), hoverinfo='none'
    )

    # Plot nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_labels, textposition='top center',
        hoverinfo='text', hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='black')),
        showlegend=False
    )

    # Legend by level
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(size=10, color=color), name=f"Level {lvl}")
        for lvl, color in level_colors.items()
    ]

    # Final figure
    fig = go.Figure(
        data=[edge_trace, node_trace] + legend_traces,
        layout=go.Layout(
            title="Hierarchical Network Graph by Level",
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    fig.show()



def plot_hierarchy_graph_plotly_by_level_and_terminal(
    graph, level_attr='level', terminal_attr='is_terminal',
    root_level=0, level_gap=2.0, x_gap=2.0
):
    """
    Plot a hierarchy graph using Plotly where:
      - Vertical placement is determined by the node's `level` attribute.
      - Marker shape is determined by the node's `is_terminal` attribute.
        * Terminal nodes: circle
        * Non-terminal nodes: square
    :param graph: nx.DiGraph
    :param level_attr: node attribute that stores the level (default: 'level')
    :param terminal_attr: node attribute that stores whether node is terminal (default: 'is_terminal')
    :param root_level: optional level to pin at the top (y=0)
    :param level_gap: vertical distance between adjacent levels
    :param x_gap: horizontal spacing between nodes within the same level
    """

    # Group nodes by level
    level_to_nodes = defaultdict(list)
    for node, data in graph.nodes(data=True):
        lvl = data.get(level_attr, 0)
        level_to_nodes[lvl].append(node)

    # Normalize levels so that root_level is at y=0 (optional)
    unique_levels = sorted(level_to_nodes.keys())
    if root_level in unique_levels:
        shift = -unique_levels.index(root_level)
        ordered_levels = list(unique_levels)
        level_index = {lvl: i + shift for i, lvl in enumerate(ordered_levels)}
    else:
        level_index = {lvl: i for i, lvl in enumerate(unique_levels)}

    # Assign colors by level
    palette = px.colors.qualitative.Light24
    level_colors = {lvl: palette[i % len(palette)] for i, lvl in enumerate(sorted(unique_levels))}

    # Build positions and split into terminal vs non-terminal
    node_positions = {}
    terminal_nodes = {"x": [], "y": [], "color": [], "size": [], "text": [], "label": []}
    non_terminal_nodes = {"x": [], "y": [], "color": [], "size": [], "text": [], "label": []}

    for lvl in sorted(level_to_nodes.keys(), key=lambda x: level_index[x]):
        nodes_at_level = level_to_nodes[lvl]
        n = len(nodes_at_level)
        xs = [x_gap * (i - (n - 1) / 2.0) for i in range(n)]
        y = -level_gap * level_index[lvl]

        for x, node in zip(xs, nodes_at_level):
            node_positions[node] = (x, y)
            deg = graph.degree(node)
            color = level_colors[lvl]
            text = f"Node {node}<br>Level: {lvl}<br>Degree: {deg}"
            label = str(node)
            is_terminal = graph.nodes[node].get(terminal_attr, False)

            if is_terminal:
                terminal_nodes["x"].append(x)
                terminal_nodes["y"].append(y)
                terminal_nodes["color"].append(color)
                terminal_nodes["size"].append(10 + 5 * deg)
                terminal_nodes["text"].append(text)
                terminal_nodes["label"].append(label)
            else:
                non_terminal_nodes["x"].append(x)
                non_terminal_nodes["y"].append(y)
                non_terminal_nodes["color"].append(color)
                non_terminal_nodes["size"].append(10 + 5 * deg)
                non_terminal_nodes["text"].append(text)
                non_terminal_nodes["label"].append(label)

    # Build edge coordinates
    edge_x, edge_y = [], []
    for src, dst in graph.edges():
        x0, y0 = node_positions[src]
        x1, y1 = node_positions[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Edges trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='gray', width=1), hoverinfo='none'
    )

    # Non-terminal nodes (square)
    non_terminal_trace = go.Scatter(
        x=non_terminal_nodes["x"], y=non_terminal_nodes["y"],
        mode='markers+text', text=non_terminal_nodes["label"], textposition='top center',
        hoverinfo='text', hovertext=non_terminal_nodes["text"],
        marker=dict(size=non_terminal_nodes["size"], color=non_terminal_nodes["color"],
                    line=dict(width=2, color='black'), symbol='square'),
        name='Non-terminal'
    )

    # Terminal nodes (circle)
    terminal_trace = go.Scatter(
        x=terminal_nodes["x"], y=terminal_nodes["y"],
        mode='markers+text', text=terminal_nodes["label"], textposition='top center',
        hoverinfo='text', hovertext=terminal_nodes["text"],
        marker=dict(size=terminal_nodes["size"], color=terminal_nodes["color"],
                    line=dict(width=2, color='black'), symbol='circle'),
        name='Terminal'
    )

    # Legend by level
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(size=10, color=color, symbol='circle'), name=f"Level {lvl}")
        for lvl, color in level_colors.items()
    ]

    # Final figure
    fig = go.Figure(
        data=[edge_trace, non_terminal_trace, terminal_trace] + legend_traces,
        layout=go.Layout(
            title="Hierarchical Network Graph by Level (Shape by Terminal Status)",
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    fig.show()
