# coding=utf-8
import pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from search_submission import PriorityQueue, a_star, bidirectional_a_star, \
    bidirectional_ucs, breadth_first_search, uniform_cost_search
from visualize_graph import plot_search


class TestPriorityQueue(unittest.TestCase):
    """Test Priority Queue implementation"""

    def test_append_and_pop(self):
        """Test the append and pop functions"""
        queue = PriorityQueue()
        temp_list = []

        for _ in xrange(10):
            a = random.randint(0, 10000)
            queue.append((a, 'a'))
            temp_list.append(a)

        temp_list = sorted(temp_list)

        for item in temp_list:
            popped = queue.pop()
            self.assertEqual(item, popped[0])


class TestBasicSearch(unittest.TestCase):
    """Test the simple search algorithms: BFS, UCS, A*"""

    def setUp(self):
        """Romania map data from Russell and Norvig, Chapter 3."""
        romania = pickle.load(open('romania_graph.pickle', 'rb'))
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()

    def test_bfs(self):
        """Test and visualize breadth-first search"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.node[n]['pos'] for n in
                          self.romania.node.keys()}

        self.romania.reset_search()
        path = breadth_first_search(self.romania, start, goal)

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path)

    def test_ucs(self):
        """TTest and visualize uniform-cost search"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.node[n]['pos'] for n in
                          self.romania.node.keys()}

        self.romania.reset_search()
        path = uniform_cost_search(self.romania, start, goal)

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path)

    def test_a_star(self):
        """Test and visualize A* search"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.node[n]['pos'] for n in
                          self.romania.node.keys()}

        self.romania.reset_search()
        path = a_star(self.romania, start, goal)

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path)

    @staticmethod
    def draw_graph(graph, node_positions=None, start=None, goal=None,
                   path=None):
        """Visualize results of graph search"""
        explored = list(graph.explored_nodes)

        labels = {}
        for node in graph:
            labels[node] = node

        if node_positions is None:
            node_positions = networkx.spring_layout(graph)

        networkx.draw_networkx_nodes(graph, node_positions)
        networkx.draw_networkx_edges(graph, node_positions, style='dashed')
        networkx.draw_networkx_labels(graph, node_positions, labels)

        networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored,
                                     node_color='g')

        if path is not None:
            edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
            networkx.draw_networkx_edges(graph, node_positions, edgelist=edges,
                                         edge_color='b')

        if start:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[start], node_color='b')

        if goal:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[goal], node_color='y')

        plt.plot()
        plt.show()


class TestBidirectionalSearch(unittest.TestCase):
    """Test the bidirectional search algorithms: UCS, A*"""

    def setUp(self):
        """Load Atlanta map data"""
        atlanta = pickle.load(open('atlanta_osm.pickle', 'rb'))
        self.atlanta = ExplorableGraph(atlanta)
        self.atlanta.reset_search()
        romania = pickle.load(open('romania_graph.pickle', 'rb'))
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()

    def test_bidirectional_ucs(self):
        """Test and generate GeoJSON for bidirectional UCS search"""
        #path = bidirectional_ucs(self.atlanta, '69581003', '69581000')
        #all_explored = self.atlanta.explored_nodes
        #plot_search(self.atlanta, 'atlanta_search_bidir_ucs.json', path,
         #           all_explored)
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.node[n]['pos'] for n in
                          self.romania.node.keys()}

        self.romania.reset_search()
        path = bidirectional_ucs(self.romania, start, goal)

    def test_bidirectional_a_star(self):
        """Test and generate GeoJSON for bidirectional A* search"""
        path = bidirectional_a_star(self.atlanta, '69581003', '69581000')
        all_explored = self.atlanta.explored_nodes
        plot_search(self.atlanta, 'atlanta_search_bidir_a_star.json', path,
                    all_explored)


if __name__ == '__main__':
    unittest.main()
