# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import pickle
import math

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
	return heapq.heappop(self.queue)


    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
	self.queue.remove(node_id)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
	heapq.heappush(self.queue, node)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """
        return key in [n for _, n in self.queue]
	
    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
	return []
    frontier = [] 
    explored = []
    frontier.append(start)
    while len(frontier) != 0:
	path = frontier.pop(0)
	eNode = path[-1]
	explored.append(eNode)
	for neigh in graph.neighbors(eNode):
		if neigh not in explored:
			nPath = list(path)
			nPath.append(neigh)
			if neigh == goal:
				return nPath
			frontier.append(nPath)					

def getFPath(pathMap, start, goal):
	path = []
	path.append(goal)
	child = goal
	while child in pathMap and pathMap[child] != start:
		parent = pathMap[child]	
		path.insert(0,parent)
		child = parent
	if (goal != start):
		path.insert(0, start)
	return path

def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
	return []
    pathCost = 0
    frontier = PriorityQueue()
    explored = []
    pathMap = {}
    frontier.append((pathCost,start))
    while frontier.size() != 0:
	path = frontier.pop()
	node = path[1]
	cost = path[0]
	if node == goal:
		"""backtrack path"""
		return getFPath(pathMap, start, goal)
	explored.append(node)
	for neigh in graph.neighbors(node):
		stepCost = graph[node][neigh]['weight']
		pathCost = cost + stepCost
		if neigh not in explored \
		and frontier.__contains__(neigh) is False:
			pathMap[neigh] = node
			frontier.append((pathCost,neigh))
		elif frontier.__contains__(neigh) is True:
			it = frontier.__iter__()
			for nodeItr in it:
				if nodeItr[1] == neigh and nodeItr[0] > pathCost:
					frontier.remove(nodeItr)
					pathMap[neigh] = node
					frontier.append((pathCost,neigh))

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node as a list.
    """
    n = graph.node[v]['pos']
    g = graph.node[goal]['pos']
    dx = abs(n[0] - g[0])
    dy = abs(n[1] - g[1])
    return math.sqrt(dx*dx + dy*dy)

def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
	return []
    frontier = PriorityQueue()
    explored = []
    g = {}
    pathMap = {}
    frontier.append((heuristic(graph, start, goal),start))
    while frontier.size() != 0:	
	path = frontier.pop()
	node = path[1]
	fCost_Node = path[0]
	hCost_Node =  heuristic(graph, node, goal)
	if node == goal:
		"""backtrack path"""
		return getFPath(pathMap, start, goal)
	explored.append(node)
	#print 'frontier : ',frontier,' explored : ',explored
	for neigh in graph.neighbors(node):
		gCost_Neigh = fCost_Node - hCost_Node + graph[node][neigh]['weight']
		#print 'parent : ',node,' neigh : ',neigh,' cost to neigh : ',gCost_Neigh, ' fCost_Node : ',fCost_Node
		if frontier.__contains__(neigh) is True:
			it = frontier.__iter__()
			for nodeItr in it:
				if nodeItr[1] == neigh and gCost_Neigh < g[neigh]:
					#print 'Updating val of - ',neigh
					frontier.remove(nodeItr)
		if neigh not in explored \
		and frontier.__contains__(neigh) is False:
			fCost_Neigh = gCost_Neigh + heuristic(graph, neigh, goal)
			g[neigh] = gCost_Neigh
			pathMap[neigh] = node
			frontier.append((fCost_Neigh,neigh))

def bidirectional_ucs(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
	return []

    pathCost = 0
    mu = float("inf")
    interNode = None

    srcFrontier = PriorityQueue()
    srcExplored = []
    srcPathMap = {}
    srcG = {}
    srcFrontier.append((pathCost,start))
    srcG[start] = pathCost

    tgtFrontier = PriorityQueue()
    tgtExplored = []
    tgtPathMap = {}
    tgtG = {}
    tgtFrontier.append((pathCost,goal))
    tgtG[goal] = pathCost

    while srcFrontier.size() != 0 and tgtFrontier.size() != 0:
	sCost = srcFrontier.top()	
	gCost = tgtFrontier.top()
	#print ' '
	#print 'srcFrontier - ',srcFrontier,' tgtFrontier - ',tgtFrontier
	if sCost[0] + gCost[0] >= mu:
		fPath = getFPath(srcPathMap, start, interNode)
		rPath = getFPath(tgtPathMap, goal, interNode)
		finalPath = fPath
		rPath = rPath[:-1]
		for rev in reversed(rPath):
			finalPath.append(rev)
		print 'finalPath - ', finalPath
		return finalPath

	# Forward Search
	fPath = srcFrontier.pop()
	fNode = fPath[1]
	fCost = fPath[0]
	srcExplored.append(fNode)
	for fNeigh in graph.neighbors(fNode):
		stepCost = graph[fNode][fNeigh]['weight']
		pathCost = fCost + stepCost
		#print ' FwdSrch : parent - ',fNode,' neigh - ',fNeigh,' stepCost - ',stepCost,' pathCost - ',pathCost
		if fNeigh in tgtExplored:
			total = pathCost + tgtG[fNeigh] 
			if total < mu:
				mu = total				
				interNode = fNeigh
			#print ' **** FwdSrch : neigh - ',fNeigh,' explored by tgt, total - ',total,' mu -',mu
		if srcFrontier.__contains__(fNeigh) is True:
			it = srcFrontier.__iter__()
			for nodeItr in it:
				if nodeItr[1] == fNeigh and nodeItr[0] > pathCost:
					srcFrontier.remove(nodeItr)
		if fNeigh not in srcExplored \
		and srcFrontier.__contains__(fNeigh) is False:
			srcPathMap[fNeigh] = fNode
			srcG[fNeigh] = pathCost
			srcFrontier.append((pathCost,fNeigh))

	# Reverse Search
	rPath = tgtFrontier.pop()
	rNode = rPath[1]
	rCost = rPath[0]
	tgtExplored.append(rNode)
	for rNeigh in graph.neighbors(rNode):
		stepCost = graph[rNode][rNeigh]['weight']
		pathCost = rCost + stepCost
		#print ' RevSrch : parent - ',rNode,' neigh - ',rNeigh,' stepCost - ',stepCost,' pathCost - ',pathCost
		if rNeigh in srcExplored:
			total = pathCost + srcG[rNeigh]
			if total < mu:
				mu = total
				interNode = rNeigh
			#print ' ***** RevSrch : neigh - ',rNeigh,' explored by src, total - ',total,' mu -',mu
		if tgtFrontier.__contains__(rNeigh) is True:
			it = tgtFrontier.__iter__()
			for nodeItr in it:
				if nodeItr[1] == rNeigh and nodeItr[0] > pathCost:
					tgtFrontier.remove(nodeItr)
		if rNeigh not in tgtExplored \
		and tgtFrontier.__contains__(rNeigh) is False:
			tgtPathMap[rNeigh] = rNode
			tgtG[rNeigh] = pathCost
			tgtFrontier.append((pathCost,rNeigh))

def pf(start,goal,node,heuristic,graph):
	pi_f = heuristic(graph, node, goal)
	pi_r = heuristic(graph, start, node)
	return 0.5*(pi_f - pi_r)

def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
	return []
    
    mu = float("inf")
    interNode = None

    srcFrontier = PriorityQueue()
    srcExplored = []
    srcPathMap = {}
    srcG = {}
    srcFrontier.append((heuristic(graph, start, goal),start))
    srcG[start] = 0

    tgtFrontier = PriorityQueue()
    tgtExplored = []
    tgtPathMap = {}
    tgtG = {}
    tgtFrontier.append((0,goal))
    tgtG[goal] = 0

    while srcFrontier.size() != 0 and tgtFrontier.size() != 0:
	sCost = srcFrontier.top()
	gCost = tgtFrontier.top()
	pr_t = -1 * pf(start, goal, goal, heuristic, graph) 
	print ' '
	print 'srcFrontier - ',srcFrontier,' tgtFrontier - ',tgtFrontier
	if sCost[0] + gCost[0] >= mu:
 		fPath = getFPath(srcPathMap, start, interNode)
                rPath = getFPath(tgtPathMap, goal, interNode)
                finalPath = fPath
                rPath = rPath[:-1]
                for rev in reversed(rPath):
                        finalPath.append(rev)
                print 'finalPath - ', finalPath
                return finalPath
	
	#Forward Search
	fPath = srcFrontier.pop()
	fNode = fPath[1]
	f_fCost_Node = fPath[0]
	f_hCost_Node = pf(start, goal, fNode, heuristic, graph)
	f_gCost_Node = srcG[fNode]
	srcExplored.append(fNode)
	for fNeigh in graph.neighbors(fNode):
		f_gCost_Neigh =  f_gCost_Node + graph[fNode][fNeigh]['weight']
		print 'FwdSrch - parent : ',fNode,' neigh : ',fNeigh,' cost to neigh : ',f_gCost_Neigh,' f_fCost_Node : ',f_fCost_Node
		if fNeigh in tgtExplored:
			total = f_gCost_Neigh + tgtG[fNeigh]
			if total < mu:	
				mu = total
				interNode = fNeigh
			print ' **** FwdSrch : neigh - ',fNeigh,' explored by tgt, total - ',total,' mu -',mu
		if srcFrontier.__contains__(fNeigh) is True:
			it = srcFrontier.__iter__()
			for nodeItr in it:
				if nodeItr[1] == fNeigh and f_gCost_Neigh < srcG[fNeigh]:
					srcFrontier.remove(nodeItr)
		if fNeigh not in srcExplored \
		and srcFrontier.__contains__(fNeigh) is False:
			f_fCost_Neigh = f_gCost_Neigh + pf(start, goal, fNeigh, heuristic, graph)
			srcG[fNeigh] = f_gCost_Neigh
			srcPathMap[fNeigh] = fNode
			srcFrontier.append((f_fCost_Neigh,fNeigh))

	# Reverse Search
	rPath = tgtFrontier.pop()
	rNode = rPath[1]
	r_fCost_Node = rPath[0]
	r_hCost_Node = -1 * pf(start, goal, rNode, heuristic, graph)
	r_GCost_Node = tgtG[rNode]
	tgtExplored.append(rNode)
	for rNeigh in graph.neighbors(rNode):
		r_gCost_Neigh = r_GCost_Node + graph[rNode][rNeigh]['weight']
		print 'RevSrch - parent : ',rNode,' neigh : ',rNeigh,' cost to neigh : ',r_gCost_Neigh,' r_fCost_Node : ',r_fCost_Node
		if rNeigh in srcExplored:
			total = r_gCost_Neigh + srcG[rNeigh]
			if total < mu:
				mu = total
				interNode = rNeigh
			print ' ***** RevSrch : neigh - ',rNeigh,' explored by src, total - ',total,' mu -',mu
		if tgtFrontier.__contains__(rNeigh) is True:
			it = tgtFrontier.__iter__()
			for nodeItr in it:
				if nodeItr[1] == rNeigh and r_gCost_Neigh < tgtG[rNeigh]:
					tgtFrontier.remove(nodeItr)
		if rNeigh not in tgtExplored \
		and tgtFrontier.__contains__(rNeigh) is False:
			r_fCost_Neigh = r_gCost_Neigh + (pf(start, goal, rNeigh, heuristic, graph) * -1)
			tgtG[rNeigh] = r_gCost_Neigh
			tgtPathMap[rNeigh] = rNode
			tgtFrontier.append((r_fCost_Neigh,rNeigh))

def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goals[0] == goals[1] and goals[0] == goals[2]:
	return [] 

    pathCost = 0
    muAB = float("inf")
    muAC = float("inf")
    muBC = float("inf")
    interNodeAB = None
    interNodeAC = None
    interNodeBC = None

    frontierA = PriorityQueue()
    exploredA = []
    pathMapA = {}
    aG = {}
    frontierA.append((pathCost,goals[0]))
    aG[goals[0]] = pathCost

    frontierB = PriorityQueue()
    exploredB = []
    pathMapB = {}
    bG = {}
    frontierB.append((pathCost,goals[1]))
    bG[goals[1]] = pathCost

    frontierC = PriorityQueue()
    exploredC = []
    pathMapC = {}
    cG = {}
    frontierC.append((pathCost,goals[2]))
    cG[goals[2]] = pathCost

    print ' '
    print 'goals[0] - ',goals[0],' goals[1] - ',goals[1],' goals[2] - ',goals[2]
    while (frontierA.size() + frontierB.size() + frontierC.size()) >= 0:
	#print ' '
	#print 'frontierA - ',frontierA,' frontierB - ',frontierB,' frontierC - ',frontierC
	#print 'exploredA - ',exploredA,' exploredB - ',exploredB,' exploredC - ',exploredC
	node1 = frontierA.top()
	node2 = frontierB.top()
	node3 = frontierC.top()

	if (node1[0] + node2[0] >= muAB) and \
	(node1[0] + node3[0] >= muAC) and \
	(node2[0] + node3[0] >= muBC):
		abFPath = getFPath(pathMapA, goals[0], interNodeAB)
		abRPath = getFPath(pathMapB, goals[1], interNodeAB)

		acFPath = getFPath(pathMapA, goals[0], interNodeAC)
		acRPath = getFPath(pathMapC, goals[2], interNodeAC)

		bcFPath = getFPath(pathMapB, goals[1], interNodeBC)
		bcRPath = getFPath(pathMapC, goals[2], interNodeBC)

		#print 'muAB : ',muAB,' muAC : ',muAC,' muBC : ',muBC
		#print 'abFPath : ',abFPath,' abRPath : ',abRPath
		#print 'acFPath : ',acFPath,' acRPath : ',acRPath
		#print 'bcFPath : ',bcFPath,' bcRPath : ',bcRPath
		maxNode = max(muAB, muAC, muBC)
		if maxNode == muAB:
			finalPath = acFPath
			acRPath = acRPath[:-1]
			for rev in reversed(acRPath):
				finalPath.append(rev)
			finalPath += bcRPath[1:]
			bcFPath = bcFPath[:-1]
			for rev in reversed(bcFPath):
				finalPath.append(rev)
		elif maxNode == muAC:
			finalPath = abFPath
			abRPath = abRPath[:-1]
			for fwd in reversed(abRPath):
				finalPath.append(fwd)
			finalPath += bcFPath[1:]
			bcRPath = bcRPath[:-1]
			for rev in reversed(bcRPath):
				finalPath.append(rev)
		else:
			finalPath = abRPath
			abFPath = abFPath[:-1] 
			for rev in reversed(abFPath):
				finalPath.append(rev)
			finalPath += acFPath[1:]
			acRPath = acRPath[:-1]
			for rev in reversed(acRPath):
				finalPath.append(rev)

		print 'finalPath : ',finalPath
		return finalPath

	# Search from A
	if frontierA.size() >= 0 and \
		(node1[0] + node2[0] < muAB or \
		node1[0] + node3[0] < muAC):

		aPath = frontierA.pop()
		aNode = aPath[1]
		aCost = aPath[0]
		exploredA.append(aNode)
		for aNeigh in graph.neighbors(aNode):
			stepCost = graph[aNode][aNeigh]['weight']
			pathCost = aCost + stepCost
			#print ' ASrch : parent - ',aNode,' neigh - ',aNeigh,' stepCost - ',stepCost,' pathCost - ',pathCost
			#print ' '
			if aNeigh in exploredB: 
				total = pathCost + bG[aNeigh]
				if total < muAB:
					muAB = total
					interNodeAB = aNeigh
					#print '**** ASrch : Convergence - muAB = ',muAB,' interNodeAB = ',interNodeAB
			if aNeigh in exploredC:
				total = pathCost + cG[aNeigh]
				if total < muAC:
					muAC = total
					interNodeAC = aNeigh
					#print '**** BSrch : Convergence - muAC = ',muAC,' interNodeAC = ',interNodeAC
			if frontierA.__contains__(aNeigh) is True:
				it = frontierA.__iter__()
				for nodeItr in it:
					if nodeItr[1] == aNeigh and nodeItr[0] > pathCost:
						frontierA.remove(nodeItr)
			if aNeigh not in exploredA \
			and frontierA.__contains__(aNeigh) is False:
				pathMapA[aNeigh] = aNode
				aG[aNeigh] = pathCost
				frontierA.append((pathCost,aNeigh))

	# Search from B
	if frontierB.size() >= 0 and \
		(node2[0] + node1[0] < muAB or \
		node2[0] + node3[0] < muBC):

		bPath = frontierB.pop()
		bNode = bPath[1]
		bCost = bPath[0]
		exploredB.append(bNode)
		for bNeigh in graph.neighbors(bNode):
			stepCost = graph[bNode][bNeigh]['weight']
			pathCost = bCost + stepCost
			#print ' BSrch : parent - ',bNode,' neigh - ',bNeigh,' stepCost - ',stepCost,' pathCost - ',pathCost
			#print ' '
			if bNeigh in exploredA: 
				total = pathCost + aG[bNeigh]
				if total < muAB:
					muAB = total
					interNodeAB = bNeigh
					#print '**** BSrch : Convergence - muAB = ',muAB,' interNodeAB = ',interNodeAB
			if bNeigh in exploredC:
				total = pathCost + cG[bNeigh]
				if total < muBC:
					muBC = total
					interNodeBC = bNeigh
					#print '**** BSrch : Convergence - muBC = ',muBC,' interNodeBC = ',interNodeBC
			if frontierB.__contains__(bNeigh) is True:
				it = frontierB.__iter__()
				for nodeItr in it:
					if nodeItr[1] == bNeigh and nodeItr[0] > pathCost:
						frontierB.remove(nodeItr)
			if bNeigh not in exploredB \
			and frontierB.__contains__(bNeigh) is False:
				pathMapB[bNeigh] = bNode
				bG[bNeigh] = pathCost
				frontierB.append((pathCost,bNeigh))

	# Search from C
	if frontierC.size() >= 0 and \
		(node3[0] + node1[0] < muAC or \
		node3[0] + node2[0] < muBC):

		cPath = frontierC.pop()
		cNode = cPath[1]
		cCost = cPath[0]
		exploredC.append(cNode)
		for cNeigh in graph.neighbors(cNode):
			stepCost = graph[cNode][cNeigh]['weight']
			pathCost = cCost + stepCost
			#print ' CSrch : parent - ',cNode,' neigh - ',cNeigh,' stepCost - ',stepCost,' pathCost - ',pathCost
			#print ' '
			if cNeigh in exploredA: 
				total = pathCost + aG[cNeigh]
				if total < muAC:
					muAC = total
					interNodeAC = cNeigh
					#print '**** CSrch : Convergence - muAC = ',muAC,' interNodeAC = ',interNodeAC
			if cNeigh in exploredB:
				total = pathCost + bG[cNeigh]
				if total < muBC:
					muBC = total
					interNodeBC = cNeigh
					#print '**** CSrch : Convergence - muBC = ',muBC,' interNodeBC = ',interNodeBC
			if frontierC.__contains__(cNeigh) is True:
				it = frontierC.__iter__()
				for nodeItr in it:
					if nodeItr[1] == cNeigh and nodeItr[0] > pathCost:
						frontierC.remove(nodeItr)
			if cNeigh not in exploredC \
			and frontierC.__contains__(cNeigh) is False:
				pathMapC[cNeigh] = cNode
				cG[cNeigh] = pathCost
				frontierC.append((pathCost,cNeigh))

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    raise NotImplementedError


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
