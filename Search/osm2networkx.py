"""
Read graphs in Open Street Maps osm format

Based on osm.py from brianw's osmgeocode
http://github.com/brianw/osmgeocode, which is based on osm.py from
comes from Graphserver:
http://github.com/bmander/graphserver/tree/master and is copyright (c)
2007, Brandon Martin-Anderson under the BSD License
"""
import sys
sys.path.append('./lib')
sys.path.append('./workspace/lib')

import xml.sax
import copy
import networkx

def download_osm(left,bottom,right,top):
    """ Return a filehandle to the downloaded data."""
    from urllib import urlopen
    fp = urlopen( "http://api.openstreetmap.org/api/0.5/map?bbox=%f,%f,%f,%f"%(left,bottom,right,top) )
    return fp

def read_osm(filename_or_stream, only_roads=True):
    """Read graph in OSM format from file specified by name or by stream object.

    Parameters
    ----------
    filename_or_stream : filename or stream object

    Returns
    -------
    G : Graph

    Examples
    --------
    >>> G=nx.read_osm(nx.download_osm(-122.33,47.60,-122.31,47.61))
    >>> plot([G.node[n]['data'].lat for n in G], [G.node[n]['data'].lon for n in G], ',')

    """
    osm = OSM(filename_or_stream)
    G = networkx.Graph()

    for w in osm.ways.itervalues():
        if only_roads and 'highway' not in w.tags:
            continue
        G.add_path(w.nds, id=w.id, data=w)
    for n_id in G.nodes_iter():
        n = osm.nodes[n_id]
        G.node[n_id] = dict(data=n)
    return G

class Node:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.pos = (lat, lon)
        self.tags = {}
   
    def __str__(self):
        return str(self.id)

    def __getitem__(self, key):
        if key == 'id':
            return self.id
        if key == 'pos':
            return (self.lon, self.lat)
        if key == 'tags':
            return self.tags

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(self.id)

class Way:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}

    def split(self, dividers):
        # slice the node-array using this nifty recursive function
        def slice_array(ar, dividers):
            for i in range(1,len(ar)-1):
                if dividers[ar[i]]>1:
                    #print "slice at %s"%ar[i]
                    left = ar[:i+1]
                    right = ar[i:]

                    rightsliced = slice_array(right, dividers)

                    return [left]+rightsliced
            return [ar]

        slices = slice_array(self.nds, dividers)

        # create a way object for each node-array slice
        ret = []
        i=0
        for slice in slices:
            littleway = copy.copy( self )
            littleway.id += "-%d"%i
            littleway.nds = slice
            ret.append( littleway )
            i += 1

        return ret

class OSM:
    def __init__(self, filename_or_stream):
        """ File can be either a filename or stream/file object."""
        nodes = {}
        ways = {}

        superself = self

        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self,loc):
                pass

            @classmethod
            def startDocument(self):
                pass

            @classmethod
            def endDocument(self):
                pass

            @classmethod
            def startElement(self, name, attrs):
                if name=='node':
                    self.currElem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']))
                elif name=='way':
                    self.currElem = Way(attrs['id'], superself)
                elif name=='tag':
                    self.currElem.tags[attrs['k']] = attrs['v']
                elif name=='nd':
                    self.currElem.nds.append( attrs['ref'] )

            @classmethod
            def endElement(self,name):
                if name=='node':
                    nodes[self.currElem.id] = self.currElem
                elif name=='way':
                    ways[self.currElem.id] = self.currElem

            @classmethod
            def characters(self, chars):
                pass

        xml.sax.parse(filename_or_stream, OSMHandler)

        self.nodes = nodes
        self.ways = ways

        #count times each node is used
        node_histogram = dict.fromkeys( self.nodes.keys(), 0 )
        for way in self.ways.values():
            if len(way.nds) < 2:       #if a way has only one node, delete it out of the osm collection
                del self.ways[way.id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1

        #use that histogram to split all ways, replacing the member set of ways
        new_ways = {}
        for id, way in self.ways.iteritems():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.id] = split_way
        self.ways = new_ways




if __name__ == '__main__':
    """Converting Atlanta OSM to pickle."""
    import math
    import random
    import pickle

    graph = read_osm('atlanta.osm')

    new_nodes = {node:graph.node[node]['data'] for node in graph.nodes()}

    euclidean = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2) * (1 + random.random())
    rads = lambda p1, p2: map(math.radians, [p1[0], p1[1], p2[0], p2[1]])
    haversine = lambda lat1, long1, lat2, long2: 6371.0 * 2*math.asin(math.sqrt(math.sin(lat2-lat1)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(long2-long1)**2))

    new_edges = [(new_nodes[node1], new_nodes[node2], haversine(*rads(new_nodes[node1]['pos'], new_nodes[node2]['pos'])))  for node1, node2 in graph.edges()]

    new_graph = networkx.Graph()
    [new_graph.add_node(node, data.__dict__) for node, data in new_nodes.items()]
    [new_graph.add_edge(s.id,t.id,weight=w) for s,t,w in new_edges]

    for key in new_graph.node.keys():
        new_graph.node[node]['pos'] = (new_graph.node[node]['lat'], new_graph.node[node]['lon'])
        new_graph.node[node]['position'] = (new_graph.node[node]['lat'], new_graph.node[node]['lon'])
    pickle.dump( new_graph , open( 'atlanta_osm.pickle' ,'wb') )

    print('Done')
