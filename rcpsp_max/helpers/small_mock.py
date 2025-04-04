import random

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import plotly.express as px
import networkx as nx

from rcpsp_max.entities.job import Job
from rcpsp_max.entities.order import Order
from rcpsp_max.entities.product import Product
from rcpsp_max.entities.resource import Resource
from rcpsp_max.entities.successors import Successor
from rcpsp_max.helpers.graphs_and_stats import gantt_resources_graphs, general_resource_usage_graph, \
    resource_usage_stats, transform_and_plot_separate
from rcpsp_max.solvers.timestamp_model import TimestampedModel
from rcpsp_max.temporal_networks.pstn_factory import get_resource_chains, RCPSP_PSTN, add_resource_chains
from temporal_networks.cstnu_tool.pstnu_to_xml import pstn_to_xml

successor = Successor(2, 12)
job1 = Job(0, 9, [successor], [5, 0])
job2 = Job(1, 5, [], [5, 0])
job3 = Job(2, 2, [], [0, 6])

resource1 = Resource(0, 5)
resource2 = Resource(1, 10)

demand1 = [0 for i in range(21)]
demand1[14] = 1

demand2 = [0 for i in range(21)]
demand2[20] = 1

product1 = Product(0, [job1, job3], demand1, 0)
product2 = Product(1, [job2], demand2, 0)

order1 = Order(0, [product1, product2], 14, 1, 20)
order2 = Order(1, [ product2], 20, 0, 20)

timestamp_model = TimestampedModel([order1, order2], [resource1, resource2], [product1, product2], 21)

res, number_of_accepted, data_df, solve_time, init_time = timestamp_model.solve_cp()

# Sort DataFrame by start time for better visualization
data_df = data_df.sort_values('start')

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis to be categorical for each task
ax.set_yticks(range(len(data_df)))
label_mapping = {
    "0_0_0": "o1p1j1",
    "1_1_1": "o2p2j2",
    "0_0_2": "o1p1j3",
    "0_1_1": "o1p2j2"
}

# Apply the mapping to set the new y-axis tick labels
ax.set_yticklabels([label_mapping.get(label, label) for label in data_df['task']])

# Plot each task as a rectangle
for i, (index, row) in enumerate(data_df.iterrows()):
    ax.add_patch(Rectangle((row['start'], i - 0.4), row['end'] - row['start'], 0.8,
                           edgecolor='black', facecolor='skyblue', alpha=0.5))

# Set the axis limits
ax.set_xlim(0, data_df['end'].max() + 1)  # +1 for visual spacing
ax.set_ylim(-1, len(data_df))

# Labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Tasks')
ax.set_title('Gantt Chart of Job Schedule')

# Grid for better readability
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# def visualize_gantt(data_df):
#     # Sort DataFrame by start time for better visualization
#     data_df = data_df.sort_values('start')
#
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(15, 8))
#
#     # Set the y-axis to be categorical for each task
#     ax.set_yticks(range(len(data_df)))
#     ax.set_yticklabels(data_df['task'])
#
#     # Assign colors based on order
#     colors = {0: 'skyblue', 1: 'lightcoral'}
#
#     # Plot each task as a rectangle
#     for i, (index, row) in enumerate(data_df.iterrows()):
#         order_id = int(row['task'].split('_')[0])  # Extract order ID from task name
#         color = colors.get(order_id, 'gray')  # Default color if order ID not in colors
#         ax.add_patch(Rectangle((row['start'], i - 0.4), row['end'] - row['start'], 0.8,
#                                edgecolor='black', facecolor=color, alpha=0.7))
#
#     # Set the axis limits
#     ax.set_xlim(0, data_df['end'].max() + 1)  # +1 for visual spacing
#     ax.set_ylim(-1, len(data_df))
#
#     # Labels and title
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Tasks')
#     ax.set_title('Job Schedule')
#
#     # Grid for better readability
#     ax.grid(axis='x', linestyle='--', alpha=0.7)
#
#     plt.tight_layout()
#     plt.show()


# Assuming data_df is already populated as per your logic
# visualize_gantt(data_df)

schedule = data_df.to_dict('records')
resource_chains, resource_assignments = get_resource_chains(schedule, timestamp_model.resources, timestamp_model.orders,
                                                            complete=True)
pstn = RCPSP_PSTN.from_rcpsp_max_instance(timestamp_model.orders)
pstn = add_resource_chains(pstn, resource_chains)

pstn_to_xml(pstn, "mock_small", "temporal_networks/cstnu_tool/xml_files")

transform_and_plot_separate(schedule, timestamp_model.orders, timestamp_model.resources)
label_mapping = {
    "0_0_0": "o1p1j1",
    "1_1_1": "o2p2j2",
    "0_0_2": "o1p1j3",
    "0_1_1": "o1p2j2"
}

# Function to replace the prefix while keeping the suffix
def translate_node(node):
    for key in label_mapping:
        if node.startswith(key):
            return node.replace(key, label_mapping[key], 1)  # Replace only the first occurrence
    return node  # Return as is if no match


def visualize_pstn(pstn):
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes
    for node in pstn.nodes:
        node_str = translate_node(pstn.translation_dict[node])
        G.add_node(node_str, label=node_str)

    # Add edges with colors and labels
    edge_labels = {}
    edge_colors = []
    edge_styles = []

    for node_from in pstn.nodes:
        for node_to in pstn.nodes:
            if node_to in pstn.edges[node_from]:
                edge = pstn.edges[node_from][node_to]
                node_from_str = translate_node(pstn.translation_dict[node_from])
                node_to_str = translate_node(pstn.translation_dict[node_to])

                if edge.weight is not None:
                    G.add_edge(node_from_str, node_to_str)
                    if edge.weight == 14 or edge.weight == 20:
                        edge_labels[(node_from_str, node_to_str)] = f"Deadline: {edge.weight}"
                        edge_colors.append("black")
                        edge_styles.append("solid")
                    else:
                        edge_labels[(node_from_str, node_to_str)] = f"Time lag: {edge.weight}"
                        edge_colors.append("blue")
                        edge_styles.append("dashed")
                else:
                    if node_from == '0_0_0_start':
                        edge_labels[(node_from_str, node_to_str)] = f"Contingent: mean = 8"
                    elif node_from == '0_1_1_start':
                        edge_labels[(node_from_str, node_to_str)] = f"Contingent: mean = 5"
                    elif node_from == '1_1_1_start':
                        edge_labels[(node_from_str, node_to_str)] = f"Contingent: mean = 5"
                    elif node_from == '0_0_2_start':
                        edge_labels[(node_from_str, node_to_str)] = f"Contingent: mean = 2"
                    edge_colors.append("red")
                    edge_styles.append("dotted")

    # Use a spring layout to spread nodes apart and reduce overlap
    pos = nx.spring_layout(G, k=2, seed=42)  # `k` controls spacing (higher = more spread out)

    # Draw graph
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=800, node_color="lightblue", font_size=10, alpha=0.9)

    # Draw edges with styles to differentiate them
    for (edge, color, style) in zip(G.edges(), edge_colors, edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color, style=style, width=2,
                               connectionstyle="arc3,rad=0.2", alpha=0.8)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    plt.title("Probabilistic Simple Temporal Network (PSTN)")
    plt.show()

visualize_pstn(pstn)

print(res)