#### Assignment 3: Probabilistic modeling

In this assignment, you will work with probabilistic models known as Bayesian networks to efficiently calculate the answer to probability questions concerning discrete random variables.

This assignment is due on October 10th, 11:59PM UTC-12. Please submit your solution as probability_solution.py.

#### Part 1 Bayesian network tutorial:

_[35 points total]_

To start, design a basic probabilistic model for the following system:

There's a nuclear power plant in which an alarm is supposed to ring when the core temperature, indicated by a gauge, exceeds a fixed threshold. For simplicity, we assume that the temperature is represented as either high or normal. However, the alarm is sometimes faulty, and the gauge is more likely to fail when the temperature is high. Use the following Boolean variables in your implementation:

> - A = alarm sounds
> - F<sub>A</sub> = alarm is faulty
> - G = gauge reading (high = True, normal = False)
> - F<sub>G</sub> = gauge is faulty
> - T = actual temperature (high = True, normal = False)

You will test your implementation at the end of the section.

#### 1a: Casting the net

_[10 points]_

Design a Bayesian network for this system, using pbnt to represent the nodes and conditional probability arcs connecting nodes. Don't worry about the probabilities for now. Fill out the function below to create the net.

The following command will create a BayesNode with 2 values, an id of 0 and the name "alarm":

    A_node = BayesNode(0,2,name='alarm')

You will use BayesNode.add\_parent() and BayesNode.add\_child() to connect nodes. For example, to connect the alarm and temperature nodes that you've already made (i.e. assuming that temperature affects the alarm probability):

    A_node.add_parent(T_node)
    T_node.add_child(A_node)
    
You can run probability\_tests.network\_setup\_test() to make sure your network is set up correctly.

#### 1b: Setting the probabilities

_[15 points]_

Assume that the following statements about the system are true:

> 1. The temperature gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty. For simplicity, say that the gauge's "true" value corresponds with its "hot" reading and "false" with its "normal" reading, so the gauge would have a 95% chance of returning "true" when the temperature is hot and it is not faulty.
> 2. The alarm is faulty 15% of the time.
> 3. The temperature is hot (call this "true") 20% of the time.
> 4. When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
> 5. The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, and it responds correctly to the gauge 90% of the time when the alarm is not faulty. For instance, when it is faulty, the alarm sounds 55% of the time that the gauge is "hot" and remains silent 55% of the time that the gauge is "normal."

Knowing these facts, set the conditional probabilities for the necessary variables on the network you just built.

Using pbnt's Distribution class: if you wanted to set the distribution for P(A) to 70% true, 30% false, you would invoke the following commands:

>     A_distribution = DiscreteDistribution(A_node)
>     index = A_distribution.generate_index([],[])
>     A_distribution[index] = [0.3,0.7]
>     A_node.set_dist(A_distribution)

Use index 0 to represent FALSE or you may run into testing issues.

If you wanted to set the distribution for P(A|G) to be

|  G  |P(A=true given G)|
| ------ | ----- |
|  T   | 0.75|
|  F   | 0.85| 

you would invoke:

    from numpy import zeros, float32
    dist = zeros([G_node.size(), A_node.size()], dtype=float32)   #Note the order of G_node, A_node
    dist[0,:] = [0.15, 0.85]
    dist[1,:] = [0.25, 0.75]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node,A_node], table=dist)
    A_node.set_dist(A_distribution)

Modeling a three-variable relationship is a bit trickier. If you wanted to set the following distribution for P(A|G,T) to be

| G   |  T  |P(A=true given G and T)|
| --- | --- |:----:|
|T|T|0.15|
|T|F|0.6|
|F|T|0.2|
|F|F|0.1|

you would invoke:

    from numpy import zeros, float32
    dist = zeros([G_node.size(), T_node.size(), A_node.size()], dtype=float32)
    dist[0,0,:] = [0.9, 0.1]
    dist[0,1,:] = [0.8, 0.2]
    dist[1,0,:] = [0.4, 0.6]
    dist[1,1,:] = [0.85, 0.15]
    
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, T_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

The key is to remember that 0 represents the index of the false probability, and 1 represents true.

You can check your probability distributions with probability_tests.probability_setup_test().

#### 1c: Probability calculations : Perform inference

_[10 points]_

To finish up, you're going to perform inference on the network to calculate the following probabilities:

> - the marginal probability that the alarm sounds
> - the marginal probability that the gauge shows "hot"
> - the probability that the temperature is actually hot, given that the alarm sounds and the alarm and gauge are both working

You'll fill out the "get_prob" functions to calculate the probabilities.

Here's an example of how to do inference for the marginal probability of the "faulty alarm" node being True (assuming "bayes_net" is your network):

    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(F_A_node)[0]
    index = Q.generate_index([True],range(Q.nDims))
    prob = Q[index]
  
To compute the conditional probability, set the evidence variables before computing the marginal as seen below (here we're computing P(A = false | F_A = true, T = False)):

    engine.evidence[F_A_node] = True
    engine.evidence[T_node] = False
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([False],range(Q.nDims))
    prob = Q[index]

If you need to sanity-check to make sure you're doing inference correctly, you can run inference on one of the probabilities that we gave you in 1b. For instance, running inference on P(T=true) should return 0.19999994 (i.e. almost 20%). You can also calculate the answers by hand to double-check.

#### Part 2: Sampling

_[65 points total]_

For the main exercise, consider the following scenario.

There are three frisbee teams who play each other: the Airheads, the Buffoons, and the Clods (A, B and C for short). 
Each match is between two teams, and each team can either win, lose, or draw in a match. Each team has a fixed but 
unknown skill level, represented as an integer from 0 to 3. The outcome of each match is probabilistically proportional to the difference in skill level between the teams.

Sampling is a method for ESTIMATING a probability distribution when it is prohibitively expensive (even for inference!) to completely compute the distribution. 

Here, we want to estimate the outcome of the matches, given prior knowledge of previous matches. Rather than using inference, we will do so by sampling the network using two [Markov Chain Monte Carlo](http://www.statistics.com/papers/LESSON1_Notes_MCMC.pdf) models: Gibbs sampling (2c) and Metropolis-Hastings (2d).

#### 2a: Build the network.

_[10 points]_

For the first sub-part, consider a network with 3 teams : the Airheads, the Buffoons, and the Clods (A, B and C for short). 3 total matches are played. 
Build a Bayes Net to represent the three teams and their influences on the match outcomes. Assume the following variable conventions:

| variable name | description|
|---------|:------:|
|A| A's skill level|
|B | B's skill level|
|C | C's skill level|
|AvB | the outcome of A vs. B <br> (0 = A wins, 1 = B wins, 2 = tie)|
|BvC | the outcome of B vs. C <br> (0 = B wins, 1 = C wins, 2 = tie)|
|CvA | the outcome of C vs. A <br> (0 = C wins, 1 = A wins, 2 = tie)|

Assume that each team has the following prior distribution of skill levels:

|skill level|P(skill level)|
|----|:----:|
|0|0.15|
|1|0.45|
|2|0.30|
|3|0.10|

In addition, assume that the differences in skill levels correspond to the following probabilities of winning:

| skill difference <br> (T2 - T1) | T1 wins | T2 wins| Tie |
|------------|----------|---|:--------:|
|0|0.10|0.10|0.80|
|1|0.20|0.60|0.20|
|2|0.15|0.75|0.10|
|3|0.05|0.90|0.05|

#### 2b: Calculate posterior distribution for the 3rd match.

_[5 points]_

Suppose that you know the following outcome of two of the three games: A beats B and A draws with C. Calculate the posterior distribution for the outcome of the BvC match in calculate_posterior(). Use the EnumerationEngine provided to perform inference- we'll be arriving at the same values by using sampling in the next section.

Hint 1: In both Metropolis-Hastings and Gibbs sampling, you'll need access to each node's probability distribution and nodes. 
You can access these by calling: 
    
    A = bayes_net.get_node_by_name("A")      
    team_table = A.dist.table
    AvB = bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    
which will return the same numpy array that you provided when constructing the probability distribution.

Hint 2: you'll also want to use the random package (e.g. random.randint()) for the probabilistic choices that sampling makes.

Hint 3: in order to count the sample states later on, you'll want to make sure the sample that you return is hashable. One way to do this is by returning the sample as a tuple.

#### 2c: Gibbs sampling
_[15 points]_

Implement the Gibbs sampling algorithm, which is a special case of Metropolis-Hastings. You'll do this in Gibbs_sampler(), which takes a Bayesian network and initial state value as a parameter and returns a sample state drawn from the network's distribution. In case of Gibbs, the returned state differs from the input state at at-most one variable (randomly chosen).

The method should just consist of a single iteration of the algorithm. If an initial value is not given, default to a state chosen uniformly at random from the possible states.

Note: DO NOT USE the given inference engines to run the sampling method, since the whole point of sampling is to calculate marginals without running inference. 


     "YOU WILL SCORE 0 POINTS ON THIS ASSIGNMENT IF YOU USE THE GIVEN INFERENCE ENGINES FOR THIS PART"


You may find [this](http://gandalf.psych.umn.edu/users/schrater/schrater_lab/courses/AI2/gibbs.pdf) helpful in understanding the basics of Gibbs sampling over Bayesian networks. 


#### 2d: Metropolis-Hastings sampling

_[15 points]_

Now you will implement the Metropolis-Hastings algorithm, which is another method for estimating a probability distribution.
The general idea of MH is to build an approximation of a latent probability distribution by repeatedly generating a "candidate" value for each random variable in the system, and then probabilistically accepting or rejecting the candidate value based on an underlying acceptance function. Unlike Gibbs, in case of MH, the returned state can differ from the initial state at more than one variable.
This [cheat sheet](http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf) provides a nice intro.

This method method should just perform a single iteration of the algorithm. If an initial value is not given, default to a state chosen uniformly at random from the possible states. 

Note: DO NOT USE the given inference engines to run the sampling method, since the whole point of sampling is to calculate marginals without running inference. 


     "YOU WILL SCORE 0 POINTS IF YOU USE THE PROVIDED INFERENCE ENGINES, OR ANY OTHER SAMPLING METHOD"

#### 2e: Comparing sampling methods

_[19 points]_

Now we are ready for the moment of truth.

Given the same outcomes as in 2b, A beats B and A draws with C, you should now estimate the likelihood of different outcomes for the third match by running Gibbs sampling until it converges to a stationary distribution. 
We'll say that the sampler has converged when, for "N" successive iterations, the difference in expected outcome for the 3rd match differs from the previous estimated outcome by less than "delta". N is a positive integer. delta goes from (0,1). For the most stationary convergence, delta should be very small. N could typically take values like 10,20,...,100 or even more.

Measure how many iterations it takes for Gibbs and MH to converge to a stationary distribution over the posterior. See for yourself how close (or not) this stable distribution is to what the Inference Engine returned in 2b. And if not, try tuning those parameters(N and delta). (You might find the concept of "burn-in" period useful). 

For the purpose of this assignment, we'll say that the sampler has converged when, for 10 successive iterations, the difference in expected outcome for the third match differs from the previous estimated outcome by less than .001 (0.1%).

Repeat this experiment for Metropolis-Hastings sampling.

Which algorithm converges more quickly? By approximately what factor? For instance, if Metropolis-Hastings takes twice as many iterations to converge as Gibbs sampling, you'd say that it converged faster by a factor of 2. Fill in sampling_question() to answer both parts.
 
####2f: Return your name

_[1 point]_

A simple task to wind down the assignment. Return you name from the function aptly called return_your_name().