# CS 6601: Artificial Intelligence - Assignment 6 - Hidden Markov Models

## Setup
Clone this repository:

`git clone https://github.gatech.edu/omscs6601/assignment_6.git`

The submission scripts depend on the presence of 3 python packages - `requests`, `future`, and `nelson`. Install them using the command below:

`pip install -r requirements.txt`

Python 2.7 is recommended and has been tested.

Read [setup.md](./setup.md) for more information on how to effectively manage your git repository and troubleshooting information.

## Overview
Hidden Markov Models are used extensively in Artificial Intelligence, Pattern Recognition, Computer Vision, and many other computer fields.  If a system has unobservable (hidden) states and each state is independent of the prior, then we can create a model of that system using probability distributions over a sequence of observations.  The idea is that we can provide this system with a series of observations to use to query what is the most likely sequence of states that generated these observations.

### Due Date
This assignment is due in Bonnie on December 3rd by 11:59PM UTC-12 (Anywhere on Earth). Note that the final exam will be released on December 4th, and the Fall Break is approaching.  However, we have allowed more time than normal for the assignment, to provide the maximum flexibility for scheduling time for preparing for the final and for family obligations. We suggest finishing this assignment quickly so as not to have too many deadlines pressing at the same time.

The deliverables for the assignment are:

* All functions completed in **_hmm_submission.py_**

### The Files

You will only have to edit and submit **_hmm_submission.py_**, but here are all the notable files:
1. **_hmm_submission.py_**: Where you were construct your probabilities and viterbi trellis.
2. **_hmm_submission_tests.py_**:  Local test file.  Due to static nature of the trellis values, local tests are extremely limited.  Please do not share values or probabilities with other students if you create your own tests.

## Submission
All submissions will be via Bonnie.  The Last Submission will be used for the Assignment grade.

enter `python submit.py` to send your submission file.  It is recommended to run this command from a shell/command prompt as issues have been known while running inside select IDEs.

### Imports
- All Standard Python modules
- Numpy

## The Assignment
Morse Code represents each letter by a unique sequence of dots and dashes. In theory, the duration of a dash is three times the duration of a dot. Each dot or dash is followed by a short silence, equal to the dot duration. The letters of a word are separated by a space equal to three dots (one dash), and the words are separated by a space equal to seven dots. The dot duration is the basic unit of time measurement in code transmission.

The chart of Morse Code looks like:

<img src="./images/morse_code.png" width=988 height=600 alt="Morse Code Chart">

### Part 1a: Encoding the HMM
_[10 Points]_

The goal here will be take a few HMMs and encode them into python.  Let's start with the Letter A as an example.

<img src="./images/A.png" alt="HMM for Letter A">

Now lets explore the components of this HMM.
* There are 4 states (A1, A2, A3, Aend) [seen in purple].
   * A1 - 1 (dot) is observed.
   * A2 - 0 (silence) is observed.
   * A3 - three 1s (dash) are observed.
   * Aend is the special state transitioned to after the HMM exits (no more letters or spaces).
* P(prior) is the probability of starting in a particular state [seen in blue].
   * In our example 'A', and for all letters, we must start with the first dot/dash state.
* P(transition) is the probability of transitioning from one state back to itself or to a new state [seen in red].
   * A dot always transitions to a silence state.
   * A dash is represented by three 1s. This is done by visiting A3 three times.  First from A2, and twice we self-transition (really stay put) to A3.
   * Silence always transitions to a dot or dash state.
* P(emission) is the probability of seeing an observation. [in green above].
   * In the case of Morse code, the output will either be a 1 (representing a dot or part of a dash) or a 0 (representing silence).

Now lets encode the above HMM into Python.

    # ['STATE1', 'STATE2', ...]

    A_states = ['A1', 'A2', 'A3', 'Aend']


    # {'STATE1': prob_of_starting_in_STATE1,
    #  'STATE2': prob_of_starting_in_STATE2, ...}

    A_prior = {'A1'  : 1.0,
               'A2'  : 0.0,
               'A3'  : 0.0,
               'Aend': 0.0}


    # {'from STATE1': {'to STATE1': prob_of_transition_from_STATE1_to_STATE1,
    #                  'to STATE2': prob_of_transition_from_STATE1_to_STATE2,
    #                  'to STATE3': prob_of_transition_from_STATE1_to_STATE3},
    # {'from STATE2': {'to STATE1': prob_of_transition_from_STATE2_to_STATE1,
    #                  'to STATE2': prob_of_transition_from_STATE2_to_STATE2,
    #                  'to STATE3': prob_of_transition_from_STATE2_to_STATE3}, ...}

    A_transition_probs = {'A1'  : {'A1'  : 0.0,
                                   'A2'  : 1.0,
                                   'A3'  : 0.0,
                                   'Aend': 0.0},
                          'A2'  : {'A1'  : 0.0,
                                   'A2'  : 0.0,
                                   'A3'  : 1.0,
                                   'Aend': 0.0},
                          'A3'  : {'A1'  : 0.0,
                                   'A2'  : 0.0,
                                   'A3'  : 0.667,
                                   'Aend': 0.333},
                          'Aend': {'A1'  : 0.0,
                                   'A2'  : 0.0,
                                   'A3'  : 0.0,
                                   'Aend': 1.0}}


    # {'STATE1': [prob_of_observing_0_in_STATE1, prob_of_observing_1_in_STATE1],
    #  'STATE2': [prob_of_observing_0_in_STATE2, prob_of_observing_1_in_STATE2], ...}

    A_emission_probs = {'A1'  : [0.0, 1.0],
                        'A2'  : [1.0, 0.0],
                        'A3'  : [0.0, 1.0],
                        'Aend': [0.0, 0.0]}

Now it's your turn.  Follow A's example, please provide the transition, prior, and emission probabilities for letters Y and Z. (accurate to 3 decimal places)

Use the following HMM probabilities:

<img src="./images/dot.png" alt="HMM for dot">
<img src="./images/dash.png" alt="HMM for dash">
<img src="./images/silence.png" alt="HMM for silence">

    Y_states = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Yend']
    Z_states = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Zend']

#### Functions to complete:
1. `part_1_a()`

---

### Part 1b: Creating the Viterbi Trellis
_[45 Points]_

The goal here will be to use a provided HMM (states, prior probabilities, transition probabilities, and emission probabilities) to build a viterbi trellis.  When provided with an evidence vector (list of observations), the function will return the most likely sequence of states that generated the evidence and the probabilty of that sequence being correct.

If no sequence can be found, the algorithm should return one of the following:
`None` (null)
`[]` (empty list)
 `['A1', 'A1', ...'A1']` (Or all being the first state)

#### Functions to complete:
1. `viterbi()`

#### Hint:
In order to reconstruct your most-likely path after running Viterbi, you'll need to keep track of a back-pointer at each state, which directs you to that state's most-likely predecessor.

#### Bonnie Notes:
In the autograder, we will also test your code against other `evidence_vectors`.

----

### Part2a: Let's add some noise!!!
_[5 Points]_

Of course it would really easy if our state outputs were always so guaranteed with a 100% probability.  To make things a little more realistic lets add some noise to our HMM.  Now with a slight change to our emission probabilities, it's possible for a state to produce the wrong output.  To make it even more difficult, I know we are monsters, we can also have transition states that don't always behave as expected.

Revisiting our example letter 'A'

<img src="./images/noise.png" alt="HMM for noisy Letter A">

And of course the new python code:

    # ['STATE1', 'STATE2', ...]

    A_states = ['A1', 'A2', 'A3', 'Aend']


    # {'STATE1': prob_of_starting_in_STATE1,
    #  'STATE2': prob_of_starting_in_STATE2, ...}

    A_prior = {'A1'  : 1.0,
               'A2'  : 0.0,
               'A3'  : 0.0,
               'Aend': 0.0}


    # {'from STATE1': {'to STATE1': prob_of_transition_from_STATE1_to_STATE1,
    #                  'to STATE2': prob_of_transition_from_STATE1_to_STATE2,
    #                  'to STATE3': prob_of_transition_from_STATE1_to_STATE3},
    # {'from STATE2': {'to STATE1': prob_of_transition_from_STATE2_to_STATE1,
    #                  'to STATE2': prob_of_transition_from_STATE2_to_STATE2,
    #                  'to STATE3': prob_of_transition_from_STATE2_to_STATE3}, ...}

    A_transition_probs = {'A1'  : {'A1'  : 0.2,
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
                                   'Aend': 0.333},
                          'Aend': {'A1'  : 0.0,
                                   'A2'  : 0.0,
                                   'A3'  : 0.0,
                                   'Aend': 1.0}}


    # {'STATE1': [prob_of_observing_0_in_STATE1, prob_of_observing_1_in_STATE1],
    #  'STATE2': [prob_of_observing_0_in_STATE2, prob_of_observing_1_in_STATE2], ...}

    A_emission_probs = {'A1'  : [0.2, 0.8],
                        'A2'  : [0.8, 0.2],
                        'A3'  : [0.2, 0.8],
                        'Aend': [0.0, 0.0]}

For this part of the assignment, note our probabilities have changed for the HMM:

<img src="./images/noisy_dot.png" alt="HMM for noisy dot">
<img src="./images/noisy_dash.png" alt="HMM for noisy dash">
<img src="./images/noisy_silence.png" alt="HMM for noisy silence">


Finally we need to be able to handle more than just single letters.  We will be adding two HMMs for spaces between letters (letter pause) and words (word space).

<img src="./images/noisy_letter_space.png" alt="HMM for noisy dot">
<img src="./images/noisy_word_space.png" alt="HMM for noisy dash">

These are encoded in Python as follows:

    # ['STATE1', 'STATE2', ...]

    letter_pause_states = ['L1']
    word_space_states = ['W1']


    # {'STATE1': prob_of_starting_in_STATE1,
    #  'STATE2': prob_of_starting_in_STATE2, ...}

    letter_pause_prior = {'L1': 1.0}
    word_space_prior = {'W1': 1.0}


    # {'from STATE1': {'to STATE1': prob_of_transition_from_STATE1_to_STATE1,
    #                  'to STATE2': prob_of_transition_from_STATE1_to_STATE2,
    #                  'to STATE3': prob_of_transition_from_STATE1_to_STATE3},
    # {'from STATE2': {'to STATE1': prob_of_transition_from_STATE2_to_STATE1,
    #                  'to STATE2': prob_of_transition_from_STATE2_to_STATE2,
    #                  'to STATE3': prob_of_transition_from_STATE2_to_STATE3}, ...}

    letter_pause_transition_probs = {'L1': {'L1': 0.667,
                                            'A1': 0.333}}
    word_space_transition_probs = {'W1': {'W1': 0.857,
                                          'A1': 0.143}}


    # {'STATE1': [prob_of_observing_0_in_STATE1, prob_of_observing_1_in_STATE1],
    #  'STATE2': [prob_of_observing_0_in_STATE2, prob_of_observing_1_in_STATE2], ...}

    letter_pause_emission_probs = {'L1': [0.8, 0.2]}
    word_space_emission_probs = {'W1' : [0.8, 0.2]}

Ok let's get started!

Suppose the `evidence_vector` [1,1,1,0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,1,1,1] is received. Only A, Y, Z, letter pause, and word space are involved. A sequence will start with a letter and end with a letter. The same letter may be repeated. The sequence can be made up of one or more words. And a word can be made up of one or more letters.

The beginning of the sequence can be any letter with equal probability. After each letter there is a letter pause or a word space, both with equal probability. A letter pause or word space can transition to any letter with equal probability.  When it reaches the end state, it stays at the end state.

Redefine the states, prior, transition and emission probabilities given above so that your Viterbi algorithm will be able to decode strings of letters. (With noise, accurate to 3 decimal places)

Finish the `quick_check` function and provide the probabilities for each based on the answers from `part_2a()`. Round to 3 decimal places.

#### Hints:
* The provided example for A only transitions to `Aend`....but all letters now can transition to both `letter_pause` and `word_space`.  We will need to adjust the current transition probability that goes from `A3` to now also include `Aend`, `letter_pause`, `word_space`.  Come to think of it, looking above we can see `letter_pause` or `word_space` only show transitions to letter A.  Hmm, (<-- see the pun), looks like our example above will need a few more updates...

#### Clarifications
Clarifications:
Please follow the examples when defining transition probabilities, emission probabilities and prior probabilities (use dictionary) in 1a.
 
Round the values to 3 decimal places for part 1 a and quick_check:
0.1234 rounds to 0.123
0.2345 rounds to 0.235
0.3456 rounds to 0.346
0.0123 rounds to 0.012
0.0125 rounds to 0.013
Those values can be hardcoded in your program. Don't use round() from python.
 
You are free to define 2a in any way you want as long as it works for your own 2b.
 
The answers of 1b and 2b should be within 10−3 of the correct answer. If no sequence can be found, the probability of 0 also needs to appear in the return.
"no sequence can be found" means the probability reaches 0 midway. If you find an incomplete sequence with some probability, output that sequence with the probability. 
Aend, Yend, .... don't appear in the answer for 1b. They are there because when I made the assignment, A3 has 0.333 probability transit to the next state. Aend is used to represent the next state. 
So in 2b, A can be followed by letter pause. Then L1 follows A3 instead of Aend.
 
If the observation is somehow shorter than the sequence, like it ends at A2 but with some probability, then the sequence should be output with that probability.
 
When output sequence in 2b, word space appears as a space. Letter pause appears as nothing.

For 2a, all priors should sum to one. Ignore the one in docstring

#### Functions to complete:
1. `part_2_a()`
2. `quick_check()`

---

### Part 2b: More Viterbi Trellis!
_[40 Points]_

Use your output in part 2a and the above `evidence_vector` to generate the most probable decoded letter sequence from the evidence sequence and its probability.

#### Functions to complete:
1. `part_2_b()`

#### Bonnie Notes:
In the autograder, we will also test your code against other evidence_vectors.

---

**CONGRATULATIONS!**  You have just completed your final assignment for CS6601 Artificial Intelligence.
