import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random as rn
from numpy.random import choice as np_choice
np.random.seed(0)

class AntColonyOptimization:

    def __init__(self, distance_matrix, num_ants, num_best, num_iterations, pheromone_decay, alpha=1, beta=1):
        """
        Parameters:
            distance_matrix (2D numpy.array): Matrix representing distances between nodes. Diagonal elements are assumed to be np.inf.
            num_ants (int): Number of ants used in each iteration.
            num_best (int): Number of best-performing ants that contribute to pheromone deposit.
            num_iterations (int): Total number of iterations to run the algorithm.
            pheromone_decay (float): Factor by which pheromone evaporates each iteration. A value less than 1 means pheromone evaporates.
            alpha (int or float): Influence of pheromone on decision-making. Higher values give more importance to pheromone.
            beta (int or float): Influence of distance on decision-making. Higher values give more weight to shorter distances.
        """
        self.distances = distance_matrix
        self.pheromone_matrix = np.ones(self.distances.shape) / len(distance_matrix)
        self.node_indices = range(len(distance_matrix))
        self.num_ants = num_ants
        self.num_best = num_best
        self.num_iterations = num_iterations
        self.pheromone_decay = pheromone_decay
        self.alpha = alpha
        self.beta = beta

    def optimize(self):
        best_path = None
        best_path_overall = ("placeholder", np.inf)
        for _ in range(self.num_iterations):
            all_paths = self.create_all_paths()
            self.update_pheromones(all_paths, self.num_best, best_path=best_path)
            best_path = min(all_paths, key=lambda x: x[1])
            print(best_path)
            if best_path[1] < best_path_overall[1]:
                best_path_overall = best_path
            self.pheromone_matrix *= self.pheromone_decay
        return best_path_overall

    def update_pheromones(self, paths, best_n, best_path):
        sorted_paths = sorted(paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:best_n]:
            for edge in path:
                self.pheromone_matrix[edge] += 1.0 / self.distances[edge]

    def calculate_path_length(self, path):
        total_distance = 0
        for edge in path:
            total_distance += self.distances[edge]
        return total_distance

    def create_all_paths(self):
        paths = []
        for _ in range(self.num_ants):
            path = self.create_path(0)
            paths.append((path, self.calculate_path_length(path)))
        return paths

    def create_path(self, start_node):
        path = []
        visited_nodes = set()
        visited_nodes.add(start_node)
        current_node = start_node
        for _ in range(len(self.distances) - 1):
            next_node = self.select_next_node(self.pheromone_matrix[current_node], self.distances[current_node], visited_nodes)
            path.append((current_node, next_node))
            current_node = next_node
            visited_nodes.add(next_node)
        path.append((current_node, start_node))  # Return to the starting node
        return path

    def select_next_node(self, pheromone, distance, visited_nodes):
        pheromone = np.copy(pheromone)
        pheromone[list(visited_nodes)] = 0

        probability = pheromone ** self.alpha * ((1.0 / distance) ** self.beta)

        normalized_prob = probability / probability.sum()
        chosen_node = np_choice(self.node_indices, 1, p=normalized_prob)[0]
        return chosen_node

# Define the size of the distance matrix
matrix_size = 5

# Initialize a distance matrix with np.inf
distance_matrix = np.full((matrix_size, matrix_size), np.inf)

# Set diagonal elements to np.inf to keep self-distances as np.inf
np.fill_diagonal(distance_matrix, np.inf)

# Function to fill the matrix with random distances
np.random.seed(0)
def populate_distances(matrix):
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            # Generate a random distance between 1 and 50
            dist = np.random.randint(1, 50)
            matrix[i, j] = dist
            matrix[j, i] = dist  # Ensure the matrix is symmetric

populate_distances(distance_matrix)

# Print the distance matrix
print(distance_matrix)

# Initialize and run the ant colony optimization
aco = AntColonyOptimization(distance_matrix, 100, 3, 50, 0.6, alpha=1, beta=1)
shortest_path_result = aco.optimize()
print("Optimal path: {}".format(shortest_path_result))

# Extract the total cost from the result
path_cost = shortest_path_result[1]

# Append the total cost to a log file
with open("path_cost_log.txt", "a") as log_file:
    log_file.write(f"{path_cost}\n")

# Create a graph visualization
graph = nx.Graph()

# Add edges with weights from the distance matrix
for i in range(len(distance_matrix)):
    for j in range(len(distance_matrix)):
        if i != j and distance_matrix[i][j] != np.inf:
            graph.add_edge(i, j, weight=distance_matrix[i][j])

# Define positions for the nodes
node_positions = nx.spring_layout(graph)

# Draw the graph with node labels
plt.figure(figsize=(10, 8))

# Draw nodes
nx.draw(graph, node_positions, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')

# Draw edges of the optimal path
optimal_path_edges = shortest_path_result[0]
nx.draw_networkx_edges(graph, node_positions, edgelist=optimal_path_edges, edge_color='red', width=2)

# Label edges of the optimal path
edge_labels = {(u, v): f'{graph[u][v]["weight"]}' for u, v in optimal_path_edges}
nx.draw_networkx_edge_labels(graph, node_positions, edge_labels=edge_labels, font_color='red')

# Remove non-optimal edges
all_edges = graph.edges()
non_optimal_edges = set(all_edges) - set(optimal_path_edges)
graph.remove_edges_from(non_optimal_edges)

# Draw the graph with only optimal path edges
plt.figure(figsize=(10, 8))
nx.draw(graph, node_positions, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
nx.draw_networkx_edges(graph, node_positions, edgelist=optimal_path_edges, edge_color='red', width=2)
nx.draw_networkx_edge_labels(graph, node_positions, edge_labels=edge_labels, font_color='red')

# Display the graph with highlighted optimal path
plt.title("Ant Colony Optimization: Optimal Path Visualization")
plt.show()
