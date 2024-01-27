import pandas as pd
from graphviz import Digraph


def visualize_tree(data, root_code):
    dot = Digraph(comment='Complaint Tree')

    def add_nodes_and_edges(code):
        # Find the row corresponding to the code
        row = data[data['Code'] == code].iloc[0]
        node_label = f"{row['Code']}: {row['Description']}"

        # Add the current node to the graph
        dot.node(str(code), node_label)

        # Find immediate children of the current node
        children = data[data['Parent'] == code]

        for _, child_row in children.iterrows():
            child_code = child_row['Code']
            child_label = f"{child_code}: {child_row['Description']}"

            # Add child node and an edge from the current node to the child
            dot.node(str(child_code), child_label)
            dot.edge(str(code), str(child_code))

            # Recursively add nodes and edges for all children
            add_nodes_and_edges(child_code)

    # Start building the tree from the root node
    add_nodes_and_edges(root_code)

    # Render the tree into a PNG file
    dot.render('tree_visualization', format='png', cleanup=True)
    return dot


data = pd.read_csv('dataset/Complaint Category.csv')
root_code = 1
dot = visualize_tree(data, root_code)

