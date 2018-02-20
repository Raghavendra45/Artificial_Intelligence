"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

#inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''

from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import random,decimal
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    A_node = BayesNode(0,2,name='alarm')
    Fa_node = BayesNode(1,2,name='faulty alarm')
    G_node = BayesNode(2,2,name='gauge')
    Fg_node = BayesNode(3,2,name='faulty gauge')
    T_node = BayesNode(4,2,name='temperature')

    # Faulty alarm and gauge reading affects the alarm
    A_node.add_parent(Fa_node)
    Fa_node.add_child(A_node)
    A_node.add_parent(G_node)
    G_node.add_child(A_node)

    # Faulty gauge and temperature affects the gauge
    G_node.add_parent(Fg_node)
    Fg_node.add_child(G_node)
    G_node.add_parent(T_node)
    T_node.add_child(G_node)

    # High temperature can cause faulty gauge
    Fg_node.add_parent(T_node)
    T_node.add_child(Fg_node)

    nodes = [A_node, Fa_node, G_node, Fg_node, T_node]

    return BayesNet(nodes)

def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")

    # P(T) = 20% and P(~T) = 80%
    T_dist = DiscreteDistribution(T_node)
    index = T_dist.generate_index([],[])
    T_dist[index] = [0.8,0.2]
    T_node.set_dist(T_dist)

    # P(Fa) = 15% and P(~Fa) = 85%
    F_A_dist = DiscreteDistribution(F_A_node)
    index = F_A_dist.generate_index([],[])
    F_A_dist[index] = [0.85,0.15]
    F_A_node.set_dist(F_A_dist)

    # P(Fg|T) = 80% and P(Fg|~T) 5%
    dist = zeros([T_node.size(),F_G_node.size()], dtype=float32)
    dist[0,:] = [0.95,0.05]
    dist[1,:] = [0.2,0.8]
    F_G_dist = ConditionalDiscreteDistribution(nodes=[T_node,F_G_node],table=dist)
    F_G_node.set_dist(F_G_dist)

    # P(G|Fg,T) = 20% and P(G|~Fg,T) = 95%
    # P(G|Fg,~T) = 80% and P(G|~Fg,~T) = 5%
    dist = zeros([F_G_node.size(),T_node.size(),G_node.size()],dtype=float32)  
    dist[0,0,:] = [0.95,0.05]
    dist[0,1,:] = [0.05,0.95]
    dist[1,0,:] = [0.2,0.8]
    dist[1,1,:] = [0.8,0.2]
    G_dist = ConditionalDiscreteDistribution(nodes=[F_G_node,T_node,G_node],table=dist)
    G_node.set_dist(G_dist)

    # P(A|Fa,G) = 55% and P(A|~Fa,G) = 90%
    # P(A|Fa,~G) = 45% and P(A|~Fa,~G) = 10%
    dist = zeros([F_A_node.size(),G_node.size(),A_node.size()],dtype=float32)
    dist[0,0,:] = [0.9,0.1]
    dist[0,1,:] = [0.1,0.9]
    dist[1,0,:] = [0.55,0.45]
    dist[1,1,:] = [0.45,0.55]
    A_dist = ConditionalDiscreteDistribution(nodes=[F_A_node,G_node,A_node],table=dist)
    A_node.set_dist(A_dist)

    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    return bayes_net

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings],range(Q.nDims))
    alarm_prob = Q[index]

    return alarm_prob

def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]  
    index = Q.generate_index([gauge_hot],range(Q.nDims))
    gauge_prob = Q[index]

    return gauge_prob

def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    T_node = bayes_net.get_node_by_name('temperature')
    A_node = bayes_net.get_node_by_name('alarm')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_G_node] = False
    engine.evidence[F_A_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot],range(Q.nDims))
    temp_prob = Q[index]

    return temp_prob

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    A_node = BayesNode(0,4,name='A')
    B_node = BayesNode(1,4,name='B')
    C_node = BayesNode(2,4,name='C')
    AvB_node = BayesNode(3,3,name='AvB')
    BvC_node = BayesNode(4,3,name='BvC')
    CvA_node = BayesNode(5,3,name='CvA')

    # Skill level of A and B affects AvB
    AvB_node.add_parent(A_node)
    AvB_node.add_parent(B_node)
    A_node.add_child(AvB_node)
    B_node.add_child(AvB_node)
    
    # Skill level of B and C affects BvC
    BvC_node.add_parent(B_node)
    BvC_node.add_parent(C_node)
    B_node.add_child(BvC_node)
    C_node.add_child(BvC_node)

    # Skill level of C and A affects BvC
    CvA_node.add_parent(A_node)
    CvA_node.add_parent(C_node)
    A_node.add_child(CvA_node)
    C_node.add_child(CvA_node)

    A_dist = DiscreteDistribution(A_node)
    index = A_dist.generate_index([],[])
    A_dist[index] = [0.15,0.45,0.30,0.10]
    A_node.set_dist(A_dist)

    B_dist = DiscreteDistribution(B_node)
    index = B_dist.generate_index([],[])
    B_dist[index] = [0.15,0.45,0.30,0.10]
    B_node.set_dist(B_dist)

    C_dist = DiscreteDistribution(C_node)
    index = C_dist.generate_index([],[])
    C_dist[index] = [0.15,0.45,0.30,0.10]
    C_node.set_dist(C_dist)

    dist = zeros([A_node.size(),B_node.size(),AvB_node.size()],dtype=float32)  
    dist[0,0,:] = [0.1,0.1,0.8]
    dist[0,1,:] = [0.2,0.6,0.2]
    dist[0,2,:] = [0.15,0.75,0.1]
    dist[0,3,:] = [0.05,0.9,0.05]

    dist[1,0,:] = [0.6,0.2,0.2]
    dist[1,1,:] = [0.1,0.1,0.8]
    dist[1,2,:] = [0.2,0.6,0.2]
    dist[1,3,:] = [0.15,0.75,0.1]
    
    dist[2,0,:] = [0.75,0.15,0.1]
    dist[2,1,:] = [0.6,0.2,0.2]
    dist[2,2,:] = [0.1,0.1,0.8]
    dist[2,3,:] = [0.2,0.6,0.2]

    dist[3,0,:] = [0.9,0.05,0.05]
    dist[3,1,:] = [0.75,0.15,0.1]
    dist[3,2,:] = [0.6,0.2,0.2]
    dist[3,3,:] = [0.1,0.1,0.8]
    AvB_dist = ConditionalDiscreteDistribution(nodes=[A_node,B_node,AvB_node],table=dist)
    AvB_node.set_dist(AvB_dist)
    # The same distribution AvB applies for BvC and CvA
    BvC_dist = ConditionalDiscreteDistribution(nodes=[B_node,C_node,BvC_node],table=dist)
    BvC_node.set_dist(BvC_dist)
    CvA_dist = ConditionalDiscreteDistribution(nodes=[C_node,A_node,CvA_node],table=dist)
    CvA_node.set_dist(CvA_dist)

    nodes = [A_node, B_node, C_node, AvB_node, BvC_node, CvA_node]

    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    BvC_node = bayes_net.get_node_by_name('BvC')
    AvB_node = bayes_net.get_node_by_name('AvB')
    CvA_node = bayes_net.get_node_by_name('CvA')
    engine = EnumerationEngine(bayes_net)
    engine.evidence[AvB_node] = 0
    engine.evidence[CvA_node] = 2 
    Q = engine.marginal(BvC_node)[0]
    index = Q.generate_index([0],range(Q.nDims))
    T1_wins = Q[index]
    index = Q.generate_index([1],range(Q.nDims))
    T2_wins = Q[index]
    index = Q.generate_index([2],range(Q.nDims))
    Tie = Q[index]

    posterior = [T1_wins,T2_wins,Tie]
    print 'calculate_posterior'
    print posterior
    sampling_question()

    return posterior # list 

def get_XRandomIndx(node,x):
    if x <= node[0]:
	return 0
    elif x > node[0] and x <= (node[0] + node[1]):
	return 1
    elif x > (node[0] + node[1]) and x <= (node[0] + node[1] + node[2]):
	return 2
    elif x > (node[0] + node[1] + node[2]) and x <= (node[0] + node[1] +node[2] + node[3]):
	return 3

def get_QRandomIndx(node,x):
    if x <= node[0]:
	return 0
    elif x > node[0] and x <= (node[0] + node[1]):
	return 1
    elif x > (node[0] + node[1]) and x <= (node[0] + node[1] + node[2]):
	return 2

def get_UnifInitState(bayes_net):
    initial_state = [0,0,0,0,0,0]
    A_node = bayes_net.get_node_by_name('A')
    A_node = A_node.dist.table
    B_node = bayes_net.get_node_by_name('B')
    B_node = B_node.dist.table
    C_node = bayes_net.get_node_by_name('C')
    C_node = C_node.dist.table
    BvC_node = bayes_net.get_node_by_name('BvC')
    BvC_node = BvC_node.dist.table
    AvB_node = bayes_net.get_node_by_name('AvB')
    AvB_node = AvB_node.dist.table
    CvA_node = bayes_net.get_node_by_name('CvA')
    CvA_node = CvA_node.dist.table

    x = random.uniform(0,1)
    initial_state[0] = get_XRandomIndx(A_node,x)     

    x = random.uniform(0,1)
    initial_state[1] = get_XRandomIndx(B_node,x)     

    x = random.uniform(0,1)
    initial_state[2] = get_XRandomIndx(C_node,x)     

    # Fix AvB to 0
    initial_state[3] = 0

    # Fix CvA to 2
    initial_state[5] = 2

    t_node = []
    sample = initial_state
    x = random.uniform(0,1)
    item0 = A_node[sample[0]] * B_node[sample[1]] * C_node[sample[2]] * AvB_node[sample[0],sample[1],sample[3]]\
			* BvC_node[sample[1],sample[2],0] * CvA_node[sample[2],sample[0],sample[5]]
    item1 = A_node[sample[0]] * B_node[sample[1]] * C_node[sample[2]] * AvB_node[sample[0],sample[1],sample[3]]\
			* BvC_node[sample[1],sample[2],1] * CvA_node[sample[2],sample[0],sample[5]]
    item2 = A_node[sample[0]] * B_node[sample[1]] * C_node[sample[2]] * AvB_node[sample[0],sample[1],sample[3]]\
			* BvC_node[sample[1],sample[2],2] * CvA_node[sample[2],sample[0],sample[5]]
    t_node.append(item0/(item0+item1+item2))
    t_node.append(item1/(item0+item1+item2))
    t_node.append(item2/(item0+item1+item2))
    initial_state[4] = get_QRandomIndx(t_node,x)

    return initial_state

def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    if len(initial_state) == 0:
	initial_state = get_UnifInitState(bayes_net)
    sample = initial_state
    A_node = bayes_net.get_node_by_name('A')
    A_node = A_node.dist.table
    B_node = bayes_net.get_node_by_name('B')
    B_node = B_node.dist.table
    C_node = bayes_net.get_node_by_name('C')
    C_node = C_node.dist.table
    BvC_node = bayes_net.get_node_by_name('BvC')
    BvC_node = BvC_node.dist.table
    AvB_node = bayes_net.get_node_by_name('AvB')
    AvB_node = AvB_node.dist.table
    CvA_node = bayes_net.get_node_by_name('CvA')
    CvA_node = CvA_node.dist.table
    t_node = []
    numbers = range(0,3) + range(4,5)
    r = random.choice(numbers)

    # Coditional probability of P(A) = P(A|MB(A)) where MB is Markov Blanket.
    # The MB of a node A includes its parents, children and other parents of all of its children.
    if r == 0:
	# The random variable to update is 'A'
 	item0 = A_node[0] * B_node[sample[1]] * C_node[sample[2]] * AvB_node[0,sample[1],sample[3]]\
				* CvA_node[sample[2],0,sample[5]]
 	item1 = A_node[1] * B_node[sample[1]] * C_node[sample[2]] * AvB_node[1,sample[1],sample[3]]\
				* CvA_node[sample[2],1,sample[5]]
 	item2 = A_node[2] * B_node[sample[1]] * C_node[sample[2]] * AvB_node[2,sample[1],sample[3]]\
				* CvA_node[sample[2],2,sample[5]]
 	item3 = A_node[3] * B_node[sample[1]] * C_node[sample[2]] * AvB_node[3,sample[1],sample[3]]\
				* CvA_node[sample[2],3,sample[5]]
        x = random.uniform(0,1)
	t_node.append(item0/(item0+item1+item2+item3))
	t_node.append(item1/(item0+item1+item2+item3))
	t_node.append(item2/(item0+item1+item2+item3))
	t_node.append(item3/(item0+item1+item2+item3))
	samp_indx = get_XRandomIndx(t_node,x)
        sample[0] = samp_indx
    elif r == 1:
	# The random variable to update is 'B'
 	item0 = A_node[sample[0]] * B_node[0] * C_node[sample[2]] * AvB_node[sample[0],0,sample[3]]\
				* BvC_node[0,sample[2],sample[4]] 
 	item1 = A_node[sample[0]] * B_node[1] * C_node[sample[2]] * AvB_node[sample[0],1,sample[3]]\
				* BvC_node[1,sample[2],sample[4]] 
 	item2 = A_node[sample[0]] * B_node[2] * C_node[sample[2]] * AvB_node[sample[0],2,sample[3]]\
				* BvC_node[2,sample[2],sample[4]] 
 	item3 = A_node[sample[0]] * B_node[3] * C_node[sample[2]] * AvB_node[sample[0],3,sample[3]]\
				* BvC_node[3,sample[2],sample[4]] 
        x = random.uniform(0,1)
	t_node.append(item0/(item0+item1+item2+item3))
	t_node.append(item1/(item0+item1+item2+item3))
	t_node.append(item2/(item0+item1+item2+item3))
	t_node.append(item3/(item0+item1+item2+item3))
	samp_indx = get_XRandomIndx(t_node,x)
        sample[1] = samp_indx
    elif r == 2:
	# The random variable to update is 'C'
 	item0 = A_node[sample[0]] * B_node[sample[1]] * C_node[0]\
				* BvC_node[sample[1],0,sample[4]] * CvA_node[0,sample[0],sample[5]]
 	item1 = A_node[sample[0]] * B_node[sample[1]] * C_node[1]\
				* BvC_node[sample[1],1,sample[4]] * CvA_node[1,sample[0],sample[5]]
 	item2 = A_node[sample[0]] * B_node[sample[1]] * C_node[2]\
				* BvC_node[sample[1],2,sample[4]] * CvA_node[2,sample[0],sample[5]]
 	item3 = A_node[sample[0]] * B_node[sample[1]] * C_node[3]\
				* BvC_node[sample[1],3,sample[4]] * CvA_node[3,sample[0],sample[5]]
        x = random.uniform(0,1)
	t_node.append(item0/(item0+item1+item2+item3))
	t_node.append(item1/(item0+item1+item2+item3))
	t_node.append(item2/(item0+item1+item2+item3))
	t_node.append(item3/(item0+item1+item2+item3))
	samp_indx = get_XRandomIndx(t_node,x)
        sample[2] = samp_indx
    elif r == 4:
	# The random variable to update is 'BvC'
 	item0 = B_node[sample[1]] * C_node[sample[2]] * BvC_node[sample[1],sample[2],0]
 	item1 = B_node[sample[1]] * C_node[sample[2]] * BvC_node[sample[1],sample[2],1]
 	item2 = B_node[sample[1]] * C_node[sample[2]] * BvC_node[sample[1],sample[2],2]
        x = random.uniform(0,1)
	t_node.append(item0/(item0+item1+item2))
	t_node.append(item1/(item0+item1+item2))
	t_node.append(item2/(item0+item1+item2))
	samp_indx = get_QRandomIndx(t_node,x)
        sample[4] = samp_indx

    sample = tuple(sample)
    return sample

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    if len(initial_state) == 0:
	initial_state = get_UnifInitState(bayes_net)
    oSample = initial_state  # Old sample for q(x(i-1))
    nSample = []             # New sample for q(xCand)
    fSample = initial_state  # Final sample based on acceptance probability 
    A_node = bayes_net.get_node_by_name('A')
    A_node = A_node.dist.table
    B_node = bayes_net.get_node_by_name('B')
    B_node = B_node.dist.table
    C_node = bayes_net.get_node_by_name('C')
    C_node = C_node.dist.table
    BvC_node = bayes_net.get_node_by_name('BvC')
    BvC_node = BvC_node.dist.table
    AvB_node = bayes_net.get_node_by_name('AvB')
    AvB_node = AvB_node.dist.table
    CvA_node = bayes_net.get_node_by_name('CvA')
    CvA_node = CvA_node.dist.table

    # In MHSampling we'll propose a candidate value for all the non-evidence variables 
    # unlike one non-evidence variable in Gibbs sampling

    # 1. Build a symmetric proposal distribution using Uniform distributions.
    t_node = []
    # Propose a candidate for 'A', 'B', 'C', 'BvC' randomly.
    cand_A = random.randint(0,3)
    cand_B = random.randint(0,3)
    cand_C = random.randint(0,3)
    cand_BvC = random.randint(0,2)
    nSample = [cand_A,cand_B,cand_C,0,cand_BvC,2] 

    # 2. Compute the probability of old and new candidate samples.
    prob_old = A_node[oSample[0]] * B_node[oSample[1]] * C_node[oSample[2]] * AvB_node[oSample[0],oSample[1],oSample[3]]\
			* BvC_node[oSample[1],oSample[2],oSample[4]] * CvA_node[oSample[2],oSample[0],oSample[5]]
    
    prob_new = A_node[nSample[0]] * B_node[nSample[1]] * C_node[nSample[2]] * AvB_node[nSample[0],nSample[1],nSample[3]]\
			* BvC_node[nSample[1],nSample[2],nSample[4]] * CvA_node[nSample[2],nSample[0],nSample[5]]

    # Accept the proposed candidate if its probability is higher than the old state
    if prob_new > prob_old:
	fSample = nSample
    else:
    	# 3. Build the Acceptance probability 'alpha'
    	alpha = min(1, prob_new / prob_old)
    	u = random.uniform(0,1)
    	if u < alpha:
		# Accept the proposed candidate
		fSample = nSample
	else:
		# Reject the proposed candidate
	        fSample = oSample
	
    sample = tuple(fSample)

    return sample

def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    G = 10000
    M = 10000

    # Perform Gibbs sampling first
    if len(initial_state) == 0:
	initial_state = get_UnifInitState(bayes_net)
    sample = initial_state
    BvC_Cnt = [0,0,0]
    nCnt = 0
    nItr = 0
    # old distribution
    pD0 = float('-inf')
    pD1 = float('-inf')
    pD2 = float('-inf')
    # current distribution
    cD0 = 0
    cD1 = 0
    cD2 = 0
    for i in range(0,500000):
	sample = Gibbs_sampler(bayes_net, sample)
	sample = list(sample)
	if sample[4] == 0:
		BvC_Cnt[0] += 1
	elif sample[4] == 1:
		BvC_Cnt[1] += 1
	elif sample[4] == 2:
		BvC_Cnt[2] += 1

	cD0 = float(BvC_Cnt[0]) / (i+1)
	cD1 = float(BvC_Cnt[1]) / (i+1)
	cD2 = float(BvC_Cnt[2]) / (i+1)
        d1 = abs(cD0 - pD0)
        d2 = abs(cD1 - pD1)
        d3 = abs(cD2 - pD2)
	if (d1 != 0 and d1 < delta) and \
		(d2 != 0 and d2 < delta) and \
		(d3 != 0 and d3 < delta):                
		if (i - nItr) == 1:
			nCnt += 1
		nItr = i

	if nCnt >= G:
		Gibbs_convergence = [cD0, cD1, cD2]
		Gibbs_count = i+1
		break
	pD0 = cD0
	pD1 = cD1
	pD2 = cD2
  
    # Perform MH sampling second
    sample = initial_state
    BvC_Cnt = [0,0,0]
    nCnt = 0
    nItr = 0
    # old distribution
    pD0 = float('-inf')
    pD1 = float('-inf')
    pD2 = float('-inf')
    # current distribution
    cD0 = 0
    cD1 = 0
    cD2 = 0
    pSample = []
    for i in range(0,500000):
	pSample = sample[:]
	sample = MH_sampler(bayes_net, sample)
	sample = list(sample)
	if sample == pSample:
		MH_rejection_count += 1
	if sample[4] == 0:
		BvC_Cnt[0] += 1
	elif sample[4] == 1:
		BvC_Cnt[1] += 1
	elif sample[4] == 2:
		BvC_Cnt[2] += 1

	cD0 = float(BvC_Cnt[0]) / (i+1)
	cD1 = float(BvC_Cnt[1]) / (i+1)
	cD2 = float(BvC_Cnt[2]) / (i+1)
        d1 = abs(cD0 - pD0)
        d2 = abs(cD1 - pD1)
        d3 = abs(cD2 - pD2)
	if (d1 != 0 and d1 < delta) and \
		(d2 != 0 and d2 < delta) and \
		(d3 != 0 and d3 < delta):                
		if (i - nItr) == 1:
			nCnt += 1
		nItr = i

	if nCnt >= M:
		MH_convergence = [cD0, cD1, cD2]
		MH_count = i+1
		break
	
	pD0 = cD0
	pD1 = cD1
	pD2 = cD2

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    gConv, mhConv, gCnt, mhCnt, mhRcnt = compare_sampling(get_game_network(),[],0.00001)
    print 'Gibbs Sample'
    print gConv
    print gCnt
    if gCnt < mhCnt:
	choice = 0
	factor = mhCnt / (gCnt * 1.0)
    else:
	choice = 1
	factor = gCnt / (mhCnt * 1.0)
  
    print 'MH Sample'
    print mhConv
    print mhCnt
    print 'MH Rejection Count : ',mhRcnt

    print 'choice : ',options[choice]
    print 'factor : ',factor

    return options[choice], factor

def return_your_name():
    """Return your name from this function"""
    return 'rmurali7'
