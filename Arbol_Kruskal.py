import networkx as nx
import matplotlib.pyplot as plt

class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, v1, v2):
        root1 = self.find(v1)
        root2 = self.find(v2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

def kruskal(graph, maximize=False):
    edges = [(weight, u, v) for u in graph for v, weight in graph[u].items()]
    edges.sort(reverse=maximize)  # Ordenar las aristas por peso (menor a mayor para MST, mayor a menor para MaxST)

    ds = DisjointSet(graph.keys())
    mst = []
    total_cost = 0

    print(f"Estado inicial:")
    print(f"Aristas ordenadas: {edges}")
    print("-" * 50)

    for weight, u, v in edges:
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst.append((u, v, weight))
            total_cost += weight

            # Imprimir el estado actual
            print(f"Añadiendo arista: ({u}, {v}) con peso {weight}")
            print(f"Árbol parcial: {mst}")
            print(f"Costo total actual: {total_cost}")
            print("-" * 50)

    return mst, total_cost

def draw_graph(graph, ax, positions, edges=None, title=""):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    nx.draw(G, positions, with_labels=True, node_color='lightblue', node_size=700, font_size=10, ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, positions, edge_labels=labels, ax=ax)

    if edges:
        edge_list = [(u, v) for u, v, weight in edges]
        nx.draw_networkx_edges(G, positions, edgelist=edge_list, edge_color='blue', width=2, ax=ax)
    
    ax.set_title(title)

# Definir el grafo como un diccionario
graph = {
    'A': {'B': 8, 'G': 9, 'H': 10, 'I': 6, 'J': 12, 'K': 3},
    'B': {'A': 8, 'C': 10, 'E': 2, 'K': 7},
    'C': {'B': 10, 'D': 9, 'K': 5},
    'D': {'C': 9, 'E': 13, 'F': 12},
    'E': {'B': 2, 'D': 13, 'F': 10},
    'F': {'D': 12, 'E': 10, 'G': 8},
    'G': {'A': 9, 'E': 6, 'F': 8, 'H': 7},
    'H': {'A': 10, 'G': 7, 'I': 3},
    'I': {'A': 6, 'H': 3, 'J': 10},
    'J': {'A': 12, 'I': 10, 'K': 8},
    'K': {'A': 3, 'B': 7, 'C': 5, 'J': 8}
}

# Definir las posiciones de los nodos en un plano 2D
positions = {
    'A': (-1, 2),
    'B': (0, 0),
    'C': (-5, 0),
    'D': (-5, -3),
    'E': (0.5, -1),
    'F': (2, -3),
    'G': (4, 1),
    'H': (3.5, 3.5),
    'I': (1.5, 5),
    'J': (-5.5, 5),
    'K': (-3, 1.5)
}

# Ejecutar el algoritmo de Kruskal para el MST (Árbol de Expansión Mínimo)
mst, total_cost_mst = kruskal(graph, maximize=False)
print("Árbol de Expansión Mínimo:", mst)
print("Costo Total del MST:", total_cost_mst)

# Ejecutar el algoritmo de Kruskal para el MaxST (Árbol de Expansión Máximo)
maxst, total_cost_maxst = kruskal(graph, maximize=True)
print("Árbol de Expansión Máximo:", maxst)
print("Costo Total del MaxST:", total_cost_maxst)

# Crear la figura y los ejes para los gráficos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

# Dibujar el grafo original
draw_graph(graph, ax1, positions, title="Grafo Original")

# Dibujar el Árbol de Expansión Mínimo
draw_graph(graph, ax2, positions, edges=mst, title="Árbol de Expansión Mínimo (Kruskal)")

# Dibujar el Árbol de Expansión Máximo
draw_graph(graph, ax3, positions, edges=maxst, title="Árbol de Expansión Máximo (Kruskal)")

# Mostrar los gráficos
plt.show()
