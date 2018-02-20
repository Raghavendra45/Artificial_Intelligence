"""Convert OSM map to GeoJSON format."""
from networkx import degree
from networkx import all_neighbors
from osm2networkx import read_osm
from random import shuffle
from random import randint
from sys import argv
from os import remove
from os.path import isfile

def get_random_nodes(graph, K=100):
    """Sample up to K random nodes from graph."""
    node_sample = graph.nodes()
    shuffle(node_sample)
    K = min(K, len(node_sample))
    node_sample = node_sample[0:K]
    return node_sample

def plot_random_nodes(graph, outfile_name, K=100):
    """Write K random nodes to GeoJSON file."""
    node_sample = get_random_nodes(graph, K)
    with open(outfile_name, 'w') as outfile:
        outfile.write('{ "type" : "FeatureCollection", \n')
        outfile.write('"features" : [\n')
        plot_nodes(node_sample, graph, outfile, False, False)
        outfile.write(']\n}')
    plot_nodes(node_sample, graph, outfile)

def plot_nodes(node_list, graph, outfile, header=False, footer=False, color="#F5A207"):
    """Write list of nodes from graph to
    GeoJSON file."""
    if header:
        outfile.write('{ "type" : "FeatureCollection", \n')
        outfile.write('"features" : [\n')
    node_strings = list(map(lambda x: node_to_GeoJSON(x, graph, color), node_list))
    outfile.write(','.join(node_strings))
    if footer:
        outfile.write(']\n}')
    print('done writing nodes to file')

def node_to_GeoJSON(node, graph, color="#F5A207"):
    """Convert node to GeoJSON string."""
    data = graph.node[node]
    lat = data['lat']
    lon = data['lon']
    node_string = ''
    node_string += '{ "type" : "Feature",\n'
    node_string += '"geometry" : {"type": "Point", "coordinates": [%f,%f]},\n'%(lon, lat)
    node_string += '"properties": {"marker-color": "%s"}\n'%(color)
    node_string += '}\n'
    return node_string

def plot_edges(edge_list, graph, outfile, header=False, footer=False, color="#F5A207"):
    """Write list of edges from graph to
    GeoJSON file."""
    if header:
        outfile.write('{ "type" : "FeatureCollection", \n')
        outfile.write('"features" : [\n')
    edge_strings = list(map(lambda x: edge_to_GeoJSON(x, graph, color), edge_list))
    outfile.write(','.join(edge_strings))
    if footer:
        outfile.write(']\n}')
    print('done writing edges to file')

def edge_to_GeoJSON(edge, graph, color="#F5A207"):
    """Convert edge to GeoJSON string."""
    start = edge[0]
    end = edge[1]
    start_lon = graph.node[start]['lon']
    start_lat = graph.node[start]['lat']
    end_lon = graph.node[end]['lon']
    end_lat = graph.node[end]['lat']
    edge_string = ''
    edge_string += '{ "type" : "Feature",\n'
    edge_string += '"geometry" : {"type": "LineString", ' 
    edge_string += '"coordinates": [[%f,%f], [%f,%f]]},\n'%(
        start_lon, start_lat, end_lon, end_lat)
    edge_string += '"properties": {"marker-color": "%s"}\n'%(color)
    edge_string += '}\n'
    return edge_string

def get_random_path(graph, K=100):
    """Pick one of top 10 most-connected nodes as 
    start, get a random neighbor, repeat up to K 
    times to build a random path.
    Returns path nodes (as ids), path edges 
    (as id tuples), all explored nodes and 
    all explored edges."""
    nodes = graph.nodes()
    nodes = sorted(nodes, key=graph.degree, reverse=True)
    start_node = nodes[randint(0,min(10,len(nodes)))]
    print('start node has degree %d'%(graph.degree(start_node)))
    path_nodes = []
    path_edges = []
    explored_nodes = set([start_node])
    last_node = start_node
    explored_nodes.add(start_node)
    K = min(K, len(nodes))
    for i in range(1,K):
        neighbors = [n for n in all_neighbors(graph, last_node)]
        new_nodes = set(neighbors).difference(set(path_nodes))
        if not new_nodes:
            print('finish with node %s with neighbors %s and explored_nodes %s'%
                (last_node, str(neighbors), str(path_nodes)))
            break
        explored_nodes = explored_nodes.union(new_nodes)
        new_nodes = sorted(list(new_nodes), key=graph.degree, reverse=True)
        neighbor = new_nodes[randint(0,min(2,len(new_nodes)-1))]
        path_edges.append((last_node, neighbor))
        last_node = neighbor
        path_nodes.append(last_node)
    return path_nodes, explored_nodes

def plot_search(graph, outfile_name, path_nodes, explored_nodes):
    """Plot path nodes/edges as well as all other 
    explored nodes in different color."""
    with open(outfile_name, 'w') as outfile:
        outfile.write('{ "type" : "FeatureCollection", \n')
        outfile.write('"features" : [\n')
        # path nodes
        plot_nodes(path_nodes, graph, outfile, False, False)
        # path edges
        outfile.write(',\n') # separate each category
        path_edges = []
        for i in range(1,len(path_nodes)):
            path_edges.append((path_nodes[i-1], path_nodes[i]))
        plot_edges(path_edges, graph, outfile, False, False)
        # explored nodes (non-path ones)
        # plot them as black
        if explored_nodes:
            explored_nodes = set(explored_nodes)
            explored_nodes = explored_nodes.difference(set(path_nodes))
            if len(explored_nodes) > 0:
                outfile.write(',\n')
                plot_nodes(explored_nodes, graph, outfile, False, False, color="#000000")
        outfile.write(']\n}')

def plot_random_path(graph, outfile_name, K=100):
    """Plot random path and write to file."""
    path_nodes, explored_nodes = get_random_path(graph, K)
    plot_search(graph, outfile_name, path_nodes, explored_nodes)
    print('done plotting random path')

def test_plot_random_path(graph, outfile_name):
    """Testing the random path plotting."""
    graph = read_osm(graph)
    if isfile(outfile_name):
        remove(outfile_name)
    plot_random_path(graph, outfile_name, K=100)

if __name__ == "__main__":
    """Read graph from file and write
    nodes/edges to file."""
    if(len(argv) > 1):
        test_plot_random_path(argv[1], argv[2])