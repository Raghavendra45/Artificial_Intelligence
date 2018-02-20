import unittest

from hmm_submission import part_1_a, viterbi, part_2_a, part_2_b, quick_check


class HMMTestsPart1(unittest.TestCase):
    """Test Part 1 of the HMM Submission
    """

    def setUp(self):
        """Setup the test probabilities.
        """
        self.y_num_states = 8
        self.z_num_states = 8
        self.num_outputs = 2

        (self.y_prior_probs,
         self.y_transition_probs,
         self.y_emission_probs,
         self.z_prior_probs,
         self.z_transition_probs,
         self.z_emission_probs) = part_1_a()

        self.a_states = ('A1', 'A2', 'A3', 'Aend')
        self.a_prior_probs = {'A1': 1.0, 'A2': 0.0, 'A3': 0.0, 'Aend': 0.0}
        self.a_transition_probs = {'A1': {'A1': 0.0,
                                          'A2': 1.0,
                                          'A3': 0.0,
                                          'Aend': 0.0},
                                   'A2': {'A1': 0.0,
                                          'A2': 0.0,
                                          'A3': 1.0,
                                          'Aend': 0.0},
                                   'A3': {'A1': 0.0,
                                          'A2': 0.0,
                                          'A3': 0.667,
                                          'Aend': 0.333},
                                   'Aend': {'A1': 0.0,
                                            'A2': 0.0,
                                            'A3': 0.0,
                                            'Aend': 1.0}}
        self.a_emission_probs = {'A1': [0.0, 1.0],
                                 'A2': [1.0, 0.0],
                                 'A3': [0.0, 1.0],
                                 'Aend': [0.0, 0.0]}

        self.evidence_vector = [1, 0, 1, 1, 1]
        self.correct_sequence = ['A1', 'A2', 'A3', 'A3', 'A3']

        self.y_states = ('Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Yend')
        self.z_states = ('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Zend')
        self.evidence_vector = [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        #self.evidence_vector = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
        self.correct_sequence = ['Y1', 'Y1', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y5', 'Y5', 'Y6', 'Y7', 'Y7', 'Y7']
	
        self.correct_probability = round(0.444889, 6)

        #self.part_1b_answer = viterbi(self.evidence_vector,
        #                              self.a_states,
        #                              self.a_prior_probs,
        #                              self.a_transition_probs,
        #                              self.a_emission_probs)
        self.part_1b_answer = viterbi(self.evidence_vector,
                                      self.y_states,
                                      self.y_prior_probs,
                                      self.y_transition_probs,
                                      self.y_emission_probs)

    def test_y_prior(self):
        """Test y prior probabilities.

        Asserts:
            Prior probabilities dictionary is correct size.
        """
        prior_probs = self.y_prior_probs
        num_states = self.y_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(prior_probs), num_states)
        self.assertTrue(len(prior_probs) is num_states, msg)

    def test_y_transition(self):
        """Test y transition probabilities.

        Asserts:
            FROM State transition probabilities dictionary is the correct sizes.
            Each TO State dictionary is the correct size.
        """
        transition_probs = self.y_transition_probs
        num_states = self.y_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(transition_probs), num_states)
        self.assertTrue(len(transition_probs) is num_states, msg)

        for key, value in transition_probs.items():
            msg = ('incorrect number of TO states FROM state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value), num_states)
            self.assertTrue(len(value) is num_states, msg)

    def test_y_emission(self):
        """Test y emission probabilities.

        Asserts:
            Emission probabilities dictionary is the correct sizes.
        """
        emission_probs = self.y_emission_probs
        num_states = self.y_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(emission_probs), num_states)
        self.assertTrue(len(emission_probs) is num_states, msg)

        for key, value in emission_probs.items():
            msg = ('incorrect number of outputs for state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value),
                                                  self.num_outputs)
            self.assertTrue(len(value) is self.num_outputs, msg)


    def test_z_prior(self):
        """Test z prior probabilities.

        Asserts:
            Prior probabilities dictionary is correct size.
        """
        prior_probs = self.z_prior_probs
        num_states = self.z_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(prior_probs), num_states)
        self.assertTrue(len(prior_probs) is num_states, msg)

    def test_z_transition(self):
        """Test z transition probabilities.

        Asserts:
            FROM State transition probabilities dictionary is the correct sizes.
            Each TO State dictionary is the correct size.
        """
        transition_probs = self.z_transition_probs
        num_states = self.z_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(transition_probs), num_states)
        self.assertTrue(len(transition_probs) is num_states, msg)

        for key, value in transition_probs.items():
            msg = ('incorrect number of TO states FROM state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value), num_states)
            self.assertTrue(len(value) is num_states, msg)

    def test_z_emission(self):
        """Test z emission probabilities.

        Asserts:
            Emission probabilities dictionary is the correct sizes.
        """
        emission_probs = self.z_emission_probs
        num_states = self.z_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(emission_probs), num_states)
        self.assertTrue(len(emission_probs) is num_states, msg)

        for key, value in emission_probs.items():
            msg = ('incorrect number of outputs for state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value),
                                                  self.num_outputs)
            self.assertTrue(len(value) is self.num_outputs, msg)

    def test_viterbi_state_sequence(self):
        """Test the output of the viterbi trellis state sequence.

        Asserts:
            Correct state sequence.
        """
        state_sequence, probability = self.part_1b_answer

        msg = ('incorrect state sequence. received {}, but expected {}'
               '').format(state_sequence, self.correct_sequence)
        self.assertTrue(state_sequence == self.correct_sequence, msg)

    def test_viterbi_probability(self):
        """Test the output of the viterbi trellis probability.

        Asserts:
            Correct probability.
        """
        state_sequence, probability = self.part_1b_answer

        msg = ('incorrect probability. received {}, but expected {}'
               '').format(round(probability, 6), self.correct_probability)
        self.assertTrue(round(probability, 6) == self.correct_probability, msg)



class HMMTestsPart2(unittest.TestCase):
    """Test Part 2 of the HMM Submission
    """

    def setUp(self):
        """Setup the test probabilities.
        """
        self.a_num_states = 4
        self.y_num_states = 8
        self.z_num_states = 8
        self.letter_pause_num_states = 1
        self.word_space_num_states = 1
        self.num_outputs = 2

        (self.a_states,
         self.a_prior_probs,
         self.a_transition_probs,
         self.a_emission_probs,
         self.y_states,
         self.y_prior_probs,
         self.y_transition_probs,
         self.y_emission_probs,
         self.z_states,
         self.z_prior_probs,
         self.z_transition_probs,
         self.z_emission_probs,
         self.letter_pause_states,
         self.letter_pause_prior_probs,
         self.letter_pause_transition_probs,
         self.letter_pause_emission_probs,
         self.word_space_states,
         self.word_space_prior_probs,
         self.word_space_transition_probs,
         self.word_space_emission_probs) = part_2_a()

        self.evidence_vector = [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
	# A A A
        #self.evidence_vector = [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
	# ZZ
        #self.evidence_vector = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
	# Y Y Y
        #self.evidence_vector = [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
	# Z Z Z
        #self.evidence_vector = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
	# A Y
        #self.evidence_vector = [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
	# A Y Z
        #self.evidence_vector = [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
        #self.evidence_vector = [1, 0]
	# Y Z
        #self.evidence_vector = [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,0, 1, 0]
	# AYZ
        self.evidence_vector = [1, 0, 1, 1, 1]

        self.correct_letter = 'A'
        self.correct_probability = round(0.0310689, 6)

        # Temp
	"""
	self.evidence_vector = [1,0,1,1,1,0,0,0,1,0,1,1,1]
	self.a_states = ['A1','A2','A3','Aend','L1']
	self.a_prior_probs = {'A1' : 0.333,
			      'A2' : 0.0,
			      'A3' : 0.0,
			    'Aend' : 0.0,
			      'L1' : 0.0}
	self.a_transition_probs = {'A1'  : {'A1'  : 0.2,
	                	            'A2'  : 0.8,
        		                    'A3'  : 0.0,
                               		    'Aend': 0.0},
	                           'A2'  : {'A1'  : 0.0,
             			            'A2'  : 0.2,
                	        	    'A3'  : 0.8,
		                            'Aend': 0.0},
	                           'A3'  : {'A1'  : 0.0,
             			            'A2'  : 0.0,
                               		    'A3'  : 0.667,
                               		    'Aend': 0.1665,
					    'L1'  : 0.1665},
                      		   'Aend': {'A1'  : 0.0,
                               		    'A2'  : 0.0,
                               		    'A3'  : 0.0,
                               		    'Aend': 1.0},
				   'L1'  : {'L1'  : 0.667,
					    'A1'  : 0.111,
					    'Y1'  : 0.111,
					    'Z1'  : 0.111}}
	self.a_emission_probs = {'A1' : [0.2,0.8],
				 'A2' : [0.8,0.2],
				 'A3' : [0.2,0.8],
				'Aend': [0.0,0.0],
				 'L1' : [0.8,0.2]}
	"""
        # Temp
        #self.states = self.a_states + self.letter_pause_states
        self.states = self.a_states
        self.trans_probs =  self.a_transition_probs.copy()
	#self.trans_probs.update(self.letter_pause_transition_probs)
	self.prior_probs = self.a_prior_probs.copy()
	#self.prior_probs.update(self.letter_pause_prior_probs)
	self.emission_probs = self.a_emission_probs.copy()
	#self.emission_probs.update(self.letter_pause_emission_probs)

	"""
        self.states = self.z_states + self.word_space_states
        self.trans_probs =  self.z_transition_probs.copy()
	self.trans_probs.update(self.word_space_transition_probs)
	self.prior_probs = self.z_prior_probs.copy()
	self.prior_probs.update(self.word_space_prior_probs)
	self.emission_probs = self.z_emission_probs.copy()
	self.emission_probs.update(self.word_space_emission_probs)

        self.states = self.a_states + self.letter_pause_states + self.word_space_states + self.y_states + self.z_states
        self.trans_probs =  self.a_transition_probs.copy()
	self.trans_probs.update(self.word_space_transition_probs)
	self.trans_probs.update(self.letter_pause_transition_probs)
	self.trans_probs.update(self.y_transition_probs)
	self.trans_probs.update(self.z_transition_probs)
	self.prior_probs = self.a_prior_probs.copy()
	self.prior_probs.update(self.word_space_prior_probs)
	self.prior_probs.update(self.letter_pause_prior_probs)
	self.prior_probs.update(self.y_prior_probs)
	self.prior_probs.update(self.z_prior_probs)
	self.emission_probs = self.a_emission_probs.copy()
	self.emission_probs.update(self.word_space_emission_probs)
	self.emission_probs.update(self.letter_pause_emission_probs)
	self.emission_probs.update(self.y_emission_probs)
	self.emission_probs.update(self.z_emission_probs)
	"""

        self.part_2b_answer = part_2_b(self.evidence_vector,
                                       self.states,
                                       self.prior_probs,
                                       self.trans_probs,
                                       self.emission_probs)


    def test_a_prior(self):
        """Test a prior probabilities.

        Asserts:
            Prior probabilities dictionary is correct size.
        """
        prior_probs = self.a_prior_probs
        num_states = self.a_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(prior_probs), num_states)
        self.assertTrue(len(prior_probs) is num_states, msg)

    def test_a_transition(self):
        """Test a transition probabilities.

        Asserts:
            FROM State transition probabilities dictionary is the correct sizes.
            Each TO State dictionary is the correct size.
        """
        transition_probs = self.a_transition_probs
        num_states = self.a_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(transition_probs), num_states)
        self.assertTrue(len(transition_probs) is num_states, msg)

        for key, value in transition_probs.items():

            last_state = num_states - 1
            state_len = num_states

            try:
                state_num = int(key[-1])
                #print state_num
                #print last_state
                if state_num == last_state:
                    state_len += 2
            except ValueError:
                pass

            msg = ('incorrect number of TO states FROM state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value), state_len)
            self.assertTrue(len(value) is state_len, msg)

    def test_a_emission(self):
        """Test a emission probabilities.

        Asserts:
            Emission probabilities dictionary is the correct sizes.
        """
        emission_probs = self.a_emission_probs
        num_states = self.a_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(emission_probs), num_states)
        self.assertTrue(len(emission_probs) is num_states, msg)

        for key, value in emission_probs.items():
            msg = ('incorrect number of outputs for state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value),
                                                  self.num_outputs)
            self.assertTrue(len(value) is self.num_outputs, msg)

    def test_y_prior(self):
        """Test y prior probabilities.

        Asserts:
            Prior probabilities dictionary is correct size.
        """
        prior_probs = self.y_prior_probs
        num_states = self.y_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(prior_probs), num_states)
        self.assertTrue(len(prior_probs) is num_states, msg)

    def test_y_transition(self):
        """Test y transition probabilities.

        Asserts:
            FROM State transition probabilities dictionary is the correct sizes.
            Each TO State dictionary is the correct size.
        """
        transition_probs = self.y_transition_probs
        num_states = self.y_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(transition_probs), num_states)
        self.assertTrue(len(transition_probs) is num_states, msg)

        for key, value in transition_probs.items():

            last_state = num_states - 1
            state_len = num_states

            try:
                state_num = int(key[-1])
                #print state_num
                #print last_state
                if state_num == last_state:
                    state_len += 2
            except ValueError:
                pass

            msg = ('incorrect number of TO states FROM state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value), state_len)
            self.assertTrue(len(value) is state_len, msg)

    def test_y_emission(self):
        """Test y emission probabilities.

        Asserts:
            Emission probabilities dictionary is the correct sizes.
        """
        emission_probs = self.y_emission_probs
        num_states = self.y_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(emission_probs), num_states)
        self.assertTrue(len(emission_probs) is num_states, msg)

        for key, value in emission_probs.items():
            msg = ('incorrect number of outputs for state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value),
                                                  self.num_outputs)
            self.assertTrue(len(value) is self.num_outputs, msg)

    def test_z_prior(self):
        """Test z prior probabilities.

        Asserts:
            Prior probabilities dictionary is correct size.
        """
        prior_probs = self.z_prior_probs
        num_states = self.z_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(prior_probs), num_states)
        self.assertTrue(len(prior_probs) is num_states, msg)

    def test_z_transition(self):
        """Test z transition probabilities.

        Asserts:
            FROM State transition probabilities dictionary is the correct sizes.
            Each TO State dictionary is the correct size.
        """
        transition_probs = self.z_transition_probs
        num_states = self.z_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(transition_probs), num_states)
        self.assertTrue(len(transition_probs) is num_states, msg)

        for key, value in transition_probs.items():

            last_state = num_states - 1
            state_len = num_states

            try:
                state_num = int(key[-1])
                if state_num == last_state:
                    state_len += 2
            except ValueError:
                pass

            msg = ('incorrect number of TO states FROM state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value), state_len)
            self.assertTrue(len(value) is state_len, msg)

    def test_z_emission(self):
        """Test z emission probabilities.

        Asserts:
            Emission probabilities dictionary is the correct sizes.
        """
        emission_probs = self.z_emission_probs
        num_states = self.z_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(emission_probs), num_states)
        self.assertTrue(len(emission_probs) is num_states, msg)

        for key, value in emission_probs.items():
            msg = ('incorrect number of outputs for state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value),
                                                  self.num_outputs)
            self.assertTrue(len(value) is self.num_outputs, msg)

    def test_letter_pause_prior(self):
        """Test letter pause prior probabilities.

        Asserts:
            Prior probabilities dictionary is correct size.
        """
        prior_probs = self.letter_pause_prior_probs
        num_states = self.letter_pause_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(prior_probs), num_states)
        self.assertTrue(len(prior_probs) is num_states, msg)

    def test_letter_pause_transition(self):
        """Test letter pause transition probabilities.

        Asserts:
            FROM State transition probabilities dictionary is the correct sizes.
            Each TO State dictionary is the correct size.
        """
        transition_probs = self.letter_pause_transition_probs
        num_states = self.letter_pause_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(transition_probs), num_states)
        self.assertTrue(len(transition_probs) is num_states, msg)

        for key, value in transition_probs.items():
            state_len = 4
            msg = ('incorrect number of TO states FROM state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value), state_len)
            self.assertTrue(len(value) is state_len, msg)

    def test_letter_pause_emission(self):
        """Test letter pause emission probabilities.

        Asserts:
            Emission probabilities dictionary is the correct sizes.
        """
        emission_probs = self.letter_pause_emission_probs
        num_states = self.letter_pause_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(emission_probs), num_states)
        self.assertTrue(len(emission_probs) is num_states, msg)

        for key, value in emission_probs.items():
            msg = ('incorrect number of outputs for state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value),
                                                  self.num_outputs)
            self.assertTrue(len(value) is self.num_outputs, msg)

    def test_word_space_prior(self):
        """Test word space prior probabilities.

        Asserts:
            Prior probabilities dictionary is correct size.
        """
        prior_probs = self.word_space_prior_probs
        num_states = self.word_space_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(prior_probs), num_states)
        self.assertTrue(len(prior_probs) is num_states, msg)

    def test_word_space_transition(self):
        """Test word space transition probabilities.

        Asserts:
            FROM State transition probabilities dictionary is the correct sizes.
            Each TO State dictionary is the correct size.
        """
        transition_probs = self.word_space_transition_probs
        num_states = self.word_space_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(transition_probs), num_states)
        self.assertTrue(len(transition_probs) is num_states, msg)

        for key, value in transition_probs.items():
            state_len = 4
            msg = ('incorrect number of TO states FROM state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value), state_len)
            self.assertTrue(len(value) is state_len, msg)

    def test_word_space_emission(self):
        """Test word space emission probabilities.

        Asserts:
            Emission probabilities dictionary is the correct sizes.
        """
        emission_probs = self.word_space_emission_probs
        num_states = self.word_space_num_states
        msg = ('incorrect number of states. counted {}, but it should be {}.'
               '').format(len(emission_probs), num_states)
        self.assertTrue(len(emission_probs) is num_states, msg)

        for key, value in emission_probs.items():
            msg = ('incorrect number of outputs for state: {}. counted {}, '
                   'but it should be {}.').format(key, len(value),
                                                  self.num_outputs)
            self.assertTrue(len(value) is self.num_outputs, msg)

    def test_part_2b_letter(self):
        """Test the output of the viterbi trellis state sequence.

        Asserts:
            Correct state sequence.
        """
        letter, probability = self.part_2b_answer

        msg = ('incorrect letter. received {}, but expected {}'
               '').format(letter, self.correct_letter)
        self.assertTrue(letter == self.correct_letter, msg)

    def test_part_2b_probability(self):
        """Test the output of the viterbi trellis probability.

        Asserts:
            Correct probability.
        """
        state_sequence, probability = self.part_2b_answer

        msg = ('incorrect probability. received {}, but expected {}'
               '').format(round(probability, 6), self.correct_probability)

        self.assertTrue(round(probability, 6) == self.correct_probability, msg)

if __name__ == '__main__':
    unittest.main(verbosity=2)
