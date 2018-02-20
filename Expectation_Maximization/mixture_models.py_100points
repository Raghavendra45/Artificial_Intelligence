from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    new_means = initial_means
    updated_image_values = image_values.copy()
    converged = False
    while converged == False:
	# 1. Get the minimum distance of each pixel to the cluster centroid
    	llist = []
        for i in range(0,len(new_means)):
		llist.append(np.sum(((image_values - new_means[i]) ** 2),axis=2))
	
	mean_inds = np.argmin(llist,axis=0)
	mean_counts = np.bincount(mean_inds.flatten())

	# 2. Compute new Cluster Centroids, which is the mean of pixels assigned to the Cluster
    	llist2 = []
    	for i in range(0,k):
		if mean_counts[i] == 0:	
			llist2.append(new_means[i])
		else:
			match_pixels = image_values[mean_inds==i]
			llist2.append(np.sum(match_pixels,axis=0)/mean_counts[i])

	prev_means = new_means
	new_means = np.array(llist2)
	
	bool_matches = (prev_means==new_means)
	if bool_matches.all() == True:
		converged = True
    		for i in range(0,k):
			updated_image_values[mean_inds==i] = new_means[i]

    return updated_image_values
	
def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
	x = val
	probs = []
	Mu = self.means
	Sigma = self.variances
	probs = (-0.5 * np.log(2*np.pi*Sigma)) - (((x-Mu)**2)/(2*Sigma))
	m = self.mixing_coefficients	
	#return np.log((m[0]*np.exp(probs[0]))+(m[1]*np.exp(probs[1]))+(m[2]*np.exp(probs[2]))+(m[3]*np.exp(probs[3]))+(m[4]*np.exp(probs[4])))
	# scipy logsumexp() equivalent for above np.log of np.exp sums
	return logsumexp(a=probs,b=m)	

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
	m = self.image_matrix.shape[0]
	n = self.image_matrix.shape[1]
	for i in range(0,self.num_components):
		x = randint(0,m-1)
		y = randint(0,n-1)
		self.means[i] = self.image_matrix[x,y]
		self.variances[i] = 1
		self.mixing_coefficients[i] = 1/self.num_components

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
	count = 0
	conv_ctr = 0
	converged = False
	flattened_image = self.image_matrix.flatten()
	reshape_image = flattened_image.reshape(flattened_image.shape[0],1)
	while converged == False:
		# 1. Get the joint log probability for all the Gaussian Component's 
		#    current mean and variances.
		Mu = self.means
		Sigma = self.variances
		m = self.mixing_coefficients	
		probs = (-0.5 * np.log(2*np.pi*Sigma)) - (((reshape_image-Mu)**2)/(2*Sigma))
		log_probs_sum = logsumexp(a=probs,b=m,axis=1)
		new_likelihood = np.sum(log_probs_sum)

		# 2. E - Step (Expectation) : Evaluate the "Responsibilities" using current parameter values
		num = m*np.exp(probs) 
		den = np.sum(num,axis=1)
		prob_Zk_Xn = num/den.reshape(den.shape[0],1)

		# 3. M - Step (Maximization) : Re-estimate the parameters using current "Responsibilities"
		Nk_div = np.sum(prob_Zk_Xn,axis=0)

		# 3a. Update Mean
		mean_K_prod = prob_Zk_Xn * reshape_image
		mean_K_sum = np.sum(mean_K_prod,axis=0)
		self.means = mean_K_sum/Nk_div

		# 3b. Update Variance
		var_K_diff = reshape_image - self.means
		var_K_diff_sqrd = var_K_diff * var_K_diff
		var_K_prod = prob_Zk_Xn * var_K_diff_sqrd
		var_K_sum = np.sum(var_K_prod,axis=0)
		self.variances = var_K_sum/Nk_div

		# 3c. Update Mixed Coefficients
		self.mixing_coefficients = Nk_div/reshape_image.shape[0]

		# 4. Check for convergence
		if count > 1:
			conv_ctr, converged = convergence_function(old_likelihood,new_likelihood, conv_ctr)

		old_likelihood = new_likelihood
		count += 1

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
	x = int(self.image_matrix.shape[0])
	y = int(self.image_matrix.shape[1])
	# Flatten the image first
	flattened_image = self.image_matrix.flatten()
	reshape_image = flattened_image.reshape(flattened_image.shape[0],1)

	# Use the trained parameters - means, variance and mixed coefficients
	Mu = self.means
	Sigma = self.variances
	m = self.mixing_coefficients	
	probs = (-0.5 * np.log(2*np.pi*Sigma)) - (((reshape_image-Mu)**2)/(2*Sigma))

	# E - Step (Expectation) : Evaluate the "Responsibilities" using trained parameter values
	num = m*np.exp(probs) 
	den = np.sum(num,axis=1)
	prob_Zk_Xn = num/den.reshape(den.shape[0],1)

	# Get the Gaussian Component for which the pixel has the max probability
	gauss_ind = np.argmax(prob_Zk_Xn,axis=1)
	gauss_ind = gauss_ind.reshape(reshape_image.shape[0],1)

	# Assign the Mean of the Gaussian Component for which the pixel has the max probability
	# as the new pixel value for image segmentation
	segment = Mu[gauss_ind]

	# Reshape the flattened image back to original dimension
	segment = segment.reshape(x,y)

	return segment

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
	flattened_image = self.image_matrix.flatten()
	reshape_image = flattened_image.reshape(flattened_image.shape[0],1)
	Mu = self.means
	Sigma = self.variances
	m = self.mixing_coefficients	
	probs = (-0.5 * np.log(2*np.pi*Sigma)) - (((reshape_image-Mu)**2)/(2*Sigma))
	log_probs_sum = logsumexp(a=probs,b=m,axis=1)
	
	return np.sum(log_probs_sum)

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
	segment_list = []
	likelihood_list = []
	for i in range(0,iters):
		self.train_model()
		segment_list.append(self.segment())
		likelihood_list.append(self.likelihood())

	return segment_list[np.argmax(likelihood_list)]

class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
	
	# 1. Use K-means clustering to get better initial means for GMM
	# 2. Get random initial centroids for K-Means
	init_means = np.zeros(self.num_components) 
	m = self.image_matrix.shape[0]
	n = self.image_matrix.shape[1]
	for i in range(0,self.num_components):
		x = randint(0,m-1)
		y = randint(0,n-1)
		# Random centroids for K-Means
		init_means[i] = self.image_matrix[x,y]
		self.variances[i] = 1
		self.mixing_coefficients[i] = 1/self.num_components

	# 3. K-means algorithm
	new_means = init_means
	image_vals = self.image_matrix.copy()
	converged = False
	while converged == False:
		# 3a. Get the min distance of each pixel to the cluster centroid
		llist = []
		for i in range(0,new_means.shape[0]):
			llist.append((image_vals - new_means[i])**2)
		
		mean_inds = np.argmin(llist,axis=0)
		mean_counts = np.bincount(mean_inds.flatten())

		# 3b. Compute new Cluster Centroids, which is the mean of pixels assigned to the Cluster
		llist2 = []
		for i in range(0,new_means.shape[0]):
			if mean_counts[i] == 0:
				llist2.append(new_means[i])
			else:		
				match_pixels = image_vals[mean_inds==i]
				llist2.append(np.sum(match_pixels,axis=0)/mean_counts[i])

		prev_means = new_means
		new_means = np.array(llist2)
		
		bool_matches = (prev_means==new_means)
		if bool_matches.all() == True:
			converged = True

	self.means = new_means

def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    low_lim = ((previous_variables * 0.99) <= new_variables)
    upp_lim = (new_variables <= (previous_variables * 1.01))

    increase_convergence_ctr = low_lim.all() & upp_lim.all()

    if increase_convergence_ctr:
	conv_ctr +=1 
    else:
	conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
	count = 0
	conv_ctr = 0
	converged = False
	flattened_image = self.image_matrix.flatten()
	reshape_image = flattened_image.reshape(flattened_image.shape[0],1)
	while converged == False:
		# 1. Get the joint log probability for all the Gaussian Component's 
		#    current mean and variances.
		Mu = self.means
		Sigma = self.variances
		m = self.mixing_coefficients	
		probs = (-0.5 * np.log(2*np.pi*Sigma)) - (((reshape_image-Mu)**2)/(2*Sigma))
		new_variables = np.array([Mu,Sigma,m])

		# 2. E - Step (Expectation) : Evaluate the "Responsibilities" using current parameter values
		num = np.exp(np.log(m) + probs)
		log_probs_sum = logsumexp(a=probs,b=m,axis=1)
		den = np.exp(log_probs_sum)
		prob_Zk_Xn = num/den.reshape(den.shape[0],1)

		# 3. M - Step (Maximization) : Re-estimate the parameters using current "Responsibilities"
		Nk_div = np.sum(prob_Zk_Xn,axis=0)

		# 3a. Update Mean
		mean_K_prod = prob_Zk_Xn * reshape_image
		mean_K_sum = np.sum(mean_K_prod,axis=0)
		self.means = np.exp(np.log(1/Nk_div) + np.log(mean_K_sum))

		# 3b. Update Variance
		var_K_diff = reshape_image - self.means
		var_K_diff_sqrd = var_K_diff * var_K_diff
		var_K_prod = prob_Zk_Xn * var_K_diff_sqrd
		var_K_sum = np.sum(var_K_prod,axis=0)
		self.variances = np.exp(np.log(1/Nk_div) + np.log(var_K_sum))

		# 3c. Update Mixed Coefficients
		self.mixing_coefficients = Nk_div/reshape_image.shape[0]

		# 4. Check for convergence
		if count > 1:
			conv_ctr, converged = convergence_function(previous_variables,new_variables, conv_ctr)

		previous_variables = new_variables
		count += 1


def bayes_info_criterion(gmm):
    # BIC = ln(n)k - 2ln(L) , where
    # k = 3 * (nos of parameters in the model. In our case, GMM has 3 parameters - means, variance and mixed coefficients) 
    # n = number of datapoints 
    # L = Likelihood ( In our case we get log likelihood using gmm.likelihood()
    likelihood = gmm.likelihood()
    m = gmm.image_matrix.shape[0]
    n = gmm.image_matrix.shape[1]
    k = 3*3
    return round((np.log(m*n) * k) - (2 * likelihood))

def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    """
    gmm_list = []
    bic_list = []
    likelihood_list = []
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    for i in range(0,6):
	num_components = i+2
	gmm = GaussianMixtureModel(image_matrix, num_components, comp_means[i])	
        gmm.initialize_training()
        gmm.train_model()
	likelihood = gmm.likelihood()
	likelihood_list.append(likelihood)
	bic = bayes_info_criterion(gmm)
	bic_list.append(bic)
	gmm_list.append(gmm)
    
    print likelihood_list
    print np.argmax(likelihood_list)
    print bic_list
    print np.argmin(bic_list)
    min_BIC_model = gmm_list[np.argmin(bic_list)]
    max_likelihood_model = gmm_list[np.argmax(likelihood_list)]
	
def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    bic = 7
    likelihood = 7
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name
    return "rmurali7"

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """
    l = points_array.shape[0]
    div = int(l/2)
    if div <= 10000:
	dist1 = np.sqrt(np.sum((points_array[0:div,np.newaxis] - means_array) ** 2,axis=2))
	dist2 = np.sqrt(np.sum((points_array[div:l,np.newaxis] - means_array) ** 2,axis=2))
	return np.vstack((dist1,dist2))
    else:
	div = int(l/3)
	dist1 = np.sqrt(np.sum((points_array[0:div,np.newaxis] - means_array) ** 2,axis=2))
	dist2 = np.sqrt(np.sum((points_array[div:2*div,np.newaxis] - means_array) ** 2,axis=2))
	dist3 = np.sqrt(np.sum((points_array[2*div:l,np.newaxis] - means_array) ** 2,axis=2))
	return np.vstack((dist1,dist2,dist3))
