from bs4 import BeautifulSoup

from temporal_networks.stnu import STNU


class PSTN(STNU):
    def __init__(self, origin_horizon=True):
        super().__init__(origin_horizon)
        self.probabilistic_durations = {}

    @classmethod
    def from_graphml(cls, file_name, origin_horizon=False) -> 'PSTN':
        """
        Create a PSTN instance from a GraphML file.
        Extracts nodes and edges, including probabilistic information from nodes.
        """
        stnu = cls(origin_horizon=origin_horizon)
        with open(file_name, 'r') as f:
            soup = BeautifulSoup(f, 'xml')

        # Add nodes and extract probabilistic parameters
        for node in soup.find_all('node'):
            node_id = node.attrs.get('id', None)
            if not node_id:
                raise ValueError("Node without id in GraphML.")

            # Add node to STNU
            stnu.add_node(node_id)

            # Extract probabilistic parameters for contingent nodes
            mean_tag = node.find(key='Mean')
            std_tag = node.find(key='StdDev')

            if mean_tag and std_tag:
                mean = float(mean_tag.text.strip())
                std = float(std_tag.text.strip())

                # Store probabilistic information for the node
                stnu.probabilistic_durations[node_id] = {"mean": mean, "std": std}

        # Process edges normally
        cls.process_graphml_edges(soup, stnu)

        return stnu

    def add_contingent_link(self, node_from: int, node_to: int, x, y, mean=None, std=None):
        """
        Add a contingent link with probabilistic parameters.

        :param node_from: Activation timepoint
        :param node_to: Contingent timepoint
        :param x: Lower bound
        :param y: Upper bound
        :param mean: Mean of the distribution
        :param std: Standard deviation of the distribution
        """
        super().add_contingent_link(node_from, node_to, x, y)
        if mean is not None and std is not None:
            self.probabilistic_durations[node_to] = {"mean": mean, "std": std}
        else:
            self.probabilistic_durations[node_to] = None
