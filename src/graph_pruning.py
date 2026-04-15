import networkx as nx
from typing import Iterable
from enum import Flag

class PruningMode(Flag):
    REMOVE_DEGREE_1 = 1
    PRUNE_PATHS = 2
    ALL = REMOVE_DEGREE_1 | PRUNE_PATHS
    
    def includes(self, other: Flag) -> bool:
        return (self & other) == other

def f1(a: int, b: int, c: int) -> int | None:
    if a == b: return c
    if a == c: return b
    return None

def prune_graph(G: nx.Graph, mode: PruningMode) -> nx.Graph:
    for a in tuple(G.nodes()):
        # v2 = Set of all nodes adjacent to "a"
        v1: Iterable[int | None] = (f1(a, b, c) for b, c in G.edges())
        v2: set[int] = set(x for x in v1 if x is not None)
        
        if mode.includes(PruningMode.REMOVE_DEGREE_1) and len(v2) == 1:
            G.remove_node(a)
        elif mode.includes(PruningMode.PRUNE_PATHS) and len(v2) == 2:
            n1, n2 = v2
            G.remove_node(a)
            G.add_edge(n1, n2)
    
    # Make sure all node labels are consecutive and start from 0
    result = nx.convert_node_labels_to_integers(G)
    
    # Assert that all node labels are consecutive and start from 0
    # (raises Exception otherwise)
    a = set(result.nodes())
    b = set(range(max(result.nodes())+1))
    if a != b:
        print("a =", tuple(sorted(a)))
        print("b =", tuple(sorted(b)))
        raise Exception(f"Unexpected error")
    
    return result

__all__ = ["PruningMode", "prune_graph"]

