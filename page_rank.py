import os
import time
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from progress import Progress

WEB_DATA = os.path.join(os.path.dirname(__file__), 'school_web.txt')


def load_graph(fd):
    """Load graph from text file

    Parameters:
    fd -- a file like object that contains lines of URL pairs

    Returns:
    A representation of the graph.

    Called for example with

    >>> graph = load_graph(open("web.txt"))

    the function parses the input file and returns a graph representation.
    Each line in the file contains two white space seperated URLs and
    denotes a directed edge (link) from the first URL to the second.
    """
    # Initiate graph dictionary
    graph = {}
    relate = {}
    # Iterate through the file line by line
    for line in fd:
        # And split each line into two URLs
        node, target = line.split()
        # Put nodes into the 'from' list
        graph.setdefault('from', [])
        # Put targets into the 'to' list
        graph.setdefault('to', [])
        graph['from'].append(node)
        graph['to'].append(target)

    # Create directional graph
    data_frame = pd.DataFrame(graph)
    G = nx.from_pandas_edgelist(data_frame, 'from', 'to', create_using=nx.DiGraph())

    nx.draw(G, arrows=True)

    # Display directional graph
    plt.show()
    return graph


def print_stats(graph):
    """Print number of nodes and edges in the given graph"""
    node_amount = len(set(graph['from']).union(set(graph['to'])))
    edge_amount = len(graph['from'])

    print(f'There are {node_amount} nodes and {edge_amount} edges in this graph.')


def stochastic_page_rank(graph, n_iter=1000000, n_steps=100):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of random walks performed
    n_steps (int) -- number of followed links before random walk is stopped

    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will after n_steps end
    on each node of the given graph.
    """

    # Set number of hits to 0 for all nodes
    hit_count = {}
    for node in graph['from']:
        hit_count.setdefault(node, 0)
    for node in graph['to']:
        hit_count.setdefault(node, 0)

    for i in range(n_iter):
        current_node = random.choice(graph['from'])
        current_node_index = graph['from'].index(current_node)
        for j in range(n_steps):
            current_node = random.choice([n for n in graph['from'] if n == graph['from'][current_node_index]])
        hit_count[current_node] += (1/n_iter)

    return hit_count


def distribution_page_rank(graph, n_iter=100):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of probability distribution updates

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """

    # Set number of hits to 1/number of nodes for all nodes
    hit_count = {}
    temp_hit_count = {}
    base_hit = 1 / len(set(graph['from']).union(set(graph['to'])))
    for node in graph['from']:
        hit_count.setdefault(node, base_hit)
        temp_hit_count.setdefault(node, 0)
    for node in graph['to']:
        hit_count.setdefault(node, base_hit)
        temp_hit_count.setdefault(node, 0)

    for i in range(n_iter):
        for node in graph['from']:
            out_degree = graph['from'].count(node)
            if out_degree == 0:
                pass
            else:
                p = 1 / out_degree
                target_indices = [graph['from'].index(n) for n in graph['from'] if n == node]
                for index in target_indices:
                    temp_hit_count[graph['to'][index]] += p
    
    return hit_count


def main():
    # Load the web structure from file
    web = load_graph(open(WEB_DATA))

    # print information about the website
    print_stats(web)

    # The graph diameter is the length of the longest shortest path
    # between any two nodes. The number of random steps of walkers
    # should be a small multiple of the graph diameter.
    diameter = 3

    # Measure how long it takes to estimate PageRank through random walks
    print("Estimate PageRank through random walks:")
    n_iter = len(web)**2
    n_steps = 2*diameter
    start = time.time()
    ranking = stochastic_page_rank(web, n_iter, n_steps)
    stop = time.time()
    time_stochastic = stop - start

    # Show top 20 pages with their page rank and time it took to compute
    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:20]))
    print(f'Calculation took {time_stochastic:.2f} seconds.\n')

    # Measure how long it takes to estimate PageRank through probabilities
    print("Estimate PageRank through probability distributions:")
    n_iter = 2*diameter
    start = time.time()
    ranking = distribution_page_rank(web, n_iter)
    stop = time.time()
    time_probabilistic = stop - start

    # Show top 20 pages with their page rank and time it took to compute
    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:20]))
    print(f'Calculation took {time_probabilistic:.2f} seconds.\n')

    # Compare the compute time of the two methods
    speedup = time_stochastic/time_probabilistic
    print(f'The probabilitic method was {speedup:.0f} times faster.')


if __name__ == '__main__':
    main()
