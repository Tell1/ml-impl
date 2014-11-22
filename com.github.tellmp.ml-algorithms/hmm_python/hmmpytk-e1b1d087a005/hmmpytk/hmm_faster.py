# hmmpytk/hmm_faster.py - HMM (Hidden Markov Model) implementation written in Python 
# Yuchen Zhang (yuchenz@cs.cmu.edu)
# Version history:
# 
# Dec 31, 2012, 0.2.0 - Big improvements in speed
# Dec 28, 2012, 0.1.1 - Added support for training on multiple instances
# Dec 27, 2012, 0.1.0 - initial version
# 
# You may distribute this software freely. 
# 
import sys
import math
import random
import copy
import pickle

class HMM:
    INF = float('inf')    # infinity !!!
    NEG_INF = float('-inf')
    M_LN2 = 0.69314718055994530942
    
    # constructor supplies state list and observation list 
    def __init__(self, states = None, observations = None, init_matrix = None, trans_matrix = None, emit_matrix = None):
        self.st_list = None         # given index, get st
        self.st_list_index = None   # given st, get index
        self.ob_list = None         # given index, get ob
        self.ob_list_index = None   # given ob, get index 
        self.init_matrix = None
        self.trans_matrix = None
        self.emit_matrix = None
        
        self.init_matrix_copy = None
        self.trans_matrix_copy = None
        self.emit_matrix_copy = None
        
        self.alpha_table = None
        self.beta_table = None
        self.xi_table = None
        self.gamma_table = None
        self.viterbi_table = None
        
        self.verbose_mode = False
                
        if (states is not None):
            self.set_states(states)
        
        if (observations is not None):
            self.set_observations(observations)        
        
        if (init_matrix is not None):
            self.set_initial_matrix(init_matrix)
        
        if (trans_matrix is not None):
            self.set_transition_matrix(trans_matrix)
        
        if (emit_matrix is not None):
            self.set_emission_matrix(emit_matrix)

    # build the empty alpha, beta, xi, gamma tables according the the length
    # of given observation sequence
    def __build_internal_tables(self, ob_len):
        st_list_len = len(self.st_list)
        st_range = xrange(st_list_len)
        ob_range = xrange(ob_len)
        neg_inf = self.NEG_INF
        
        self.alpha_table = [[neg_inf for st in st_range] for t in xrange(ob_len + 1)]
        self.beta_table = [[neg_inf for st in st_range] for t in xrange(ob_len + 1)]
        self.xi_table = [[[neg_inf for st_j in st_range] for st_i in st_range] for ob in ob_range]
        self.gamma_table = [[neg_inf for st in st_range] for ob in ob_range]

    # calculate log(exp(left) + exp(right)) more accurately
    # based on http://www.cs.cmu.edu/~roni/11761-s12/assignments/log_add.c
    def __log_add(self, left, right):
        if (right < left):
            return left + math.log1p(math.exp(right - left))
        elif (right > left):
            return right + math.log1p(math.exp(left - right))
        else:
            return left + self.M_LN2

    # calculate __ln(x)
    def __ln(self, value):
        if (value == 0.0):
            return self.NEG_INF
        else:
            return math.log(value)
    
    # given the observation sequence, return the most probable state sequence
    def viterbi(self, ob_seq):
        N = len(self.st_list)
        ob_seq_len = len(ob_seq)
        
        viterbi_table = [[self.NEG_INF for st in xrange(N)] for t in xrange(ob_seq_len + 1)]
        bp_table = [[self.NEG_INF for st in xrange(N)] for t in xrange(ob_seq_len)]
        ob_seq_int = [self.ob_list_index[ob] for ob in ob_seq]

        # initialize viterbi table's first entry
        for i in xrange(N):
            viterbi_table[0][i] = self.init_matrix[i]
        
        for t in xrange(1, ob_seq_len + 1): # loop through time
            ot = ob_seq_int[t - 1]
            for st_j in xrange(N): # for each state
                viterbi_t_j = self.NEG_INF
                curr_best_st = None
                
                for st_i in xrange(N):
                    log_sum = viterbi_table[t - 1][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot]

                    if (log_sum > viterbi_t_j):
                        viterbi_t_j = log_sum
                        curr_best_st = st_i

                viterbi_table[t][st_j] = viterbi_t_j
                bp_table[t - 1][st_j] = curr_best_st
        
        # perform the viterbi back-trace
        t = len(bp_table) - 1
        if (t < 0):
            return None
        states_seq = [0] * ob_seq_len
        
        curr_state = 0
        for st in xrange(N):
            if (bp_table[-1][st] > bp_table[-1][curr_state]):
                curr_state = st
        
        for t in xrange(ob_seq_len - 1, -1, -1):
            if (curr_state is None):
                return None
            states_seq[t] = self.st_list[curr_state]
            curr_state = bp_table[t][curr_state]
        
        return states_seq
    
    # given the observation sequence, return its probability given the model
    def forward(self, ob_seq_int):
        if (self.verbose_mode):
            sys.stderr.write("\nComputing Alpha table ... \n")
            
        N = len(self.st_list)
        ob_seq_len = len(ob_seq_int)
        
        # build the alpha table in advance if not compatible
        if (self.alpha_table is None or len(self.alpha_table) != (ob_seq_len + 1)  
            or len(self.alpha_table[0]) != N):
            self.alpha_table = [[0 for st in xrange(N)] for ob in xrange(ob_seq_len + 1)]
        
        for i in xrange(N):
            self.alpha_table[0][i] = self.init_matrix[i]

        for t in xrange(1, ob_seq_len + 1): # loop through time
            if (self.verbose_mode):
                sys.stderr.write("\rComputing alpha table t = %d out of %d"%(t, len(ob_seq_int)))  
                          
            ot = ob_seq_int[t - 1]
            for st_j in xrange(N):
                alpha_t_j = self.NEG_INF
                for st_i in xrange(N):
                    log_sum = self.alpha_table[t - 1][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot]
                    alpha_t_j = self.__log_add(alpha_t_j, log_sum)

                self.alpha_table[t][st_j] = alpha_t_j
    
    # backward algorithm
    def backward(self, ob_seq_int):
        if (self.verbose_mode):
            sys.stderr.write("\nComputing Beta table ... \n")
            
        N = len(self.st_list)
        ob_seq_len = len(ob_seq_int)
        
        # build the beta table in advance if not compatible
        if (self.beta_table is None or len(self.beta_table) != (ob_seq_len + 1) 
            or len(self.beta_table[0]) != N):
            self.beta_table = [[0 for st in xrange(N)] for ob in xrange(ob_seq_len + 1)]
            
        # create the beta table in advance since we are filling backwards
        ln_1 = self.__ln(1.0)
        ln_0 = self.NEG_INF
        for st in xrange(N):
            self.beta_table[-1][st] = ln_1
        
        for t in xrange(ob_seq_len - 1):
            for st in xrange(N):
                self.beta_table[t][st] = ln_0
    
        # starting from 2nd to last and move backwards
        for t in xrange(ob_seq_len - 1, -1, -1):
            if (self.verbose_mode):
                sys.stderr.write("\rComputing beta table t = %d out of %d"%(t, len(ob_seq_int)))
                            
            ot_next = ob_seq_int[t]
            for st_i in xrange(N):
                beta_t_i = self.NEG_INF
                for st_j in xrange(N):
                    log_sum = self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot_next] + self.beta_table[t + 1][st_j]
                    beta_t_i = self.__log_add(beta_t_i, log_sum)
                    # j += 1
    
                self.beta_table[t][st_i] = beta_t_i

    # computes the __Xi[t][i][j]: being at st[i] at time t, and at st[j] at time t + 1
    def __Xi(self, st_i, st_j, t, ob_seq_int):
        ot_next = ob_seq_int[t + 1]
        
        numerator = self.alpha_table[t + 1][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot_next] + self.beta_table[t + 1][st_j]
        denominator = self.__ln(0.0)
        
        N = len(self.st_list)
        for st_from in xrange(N):
            for st_to in xrange(N):
                denominator = self.__log_add(denominator, self.alpha_table[t + 1][st_from] + self.trans_matrix[st_from][st_to] + self.emit_matrix[st_to][ot_next] + self.beta_table[t + 1][st_to])

        return (numerator - denominator) 

    # get average likelihood from alpha table
    def __get_avgll_by_alpha(self, ob_seq_len):
        log_sum = self.NEG_INF
        for st in xrange(len(self.st_list)):
            log_sum = self.__log_add(log_sum, self.alpha_table[-1][st])
        
        return (log_sum / float(ob_seq_len))

    # train the HMM using forward-backward algorithm, stops when the differences 
    # between likelihoods is smaller than delta, or reaches maximum iteration
    def train(self, ob_seq, max_iteration = 100, delta = 0.000001):
        if (max_iteration < 1):
            return None

        iteration = 0
        prev_avg_ll = self.NEG_INF
        ob_seq_int = [self.ob_list_index[ob] for ob in ob_seq]
        ob_seq_len = len(ob_seq)
        
        # build all the matrices
        self.__build_internal_tables(ob_seq_len)
        
        while (iteration < max_iteration):
            self.forward(ob_seq_int)
            self.backward(ob_seq_int)
            avg_ll = self.__get_avgll_by_alpha(len(ob_seq))
            
            self.forward_backward(ob_seq_int)
            
            # swap the matrices
            self.init_matrix, self.init_matrix_copy = self.init_matrix_copy, self.init_matrix
            self.trans_matrix, self.trans_matrix_copy = self.trans_matrix_copy, self.trans_matrix
            self.emit_matrix, self.emit_matrix_copy = self.emit_matrix_copy, self.emit_matrix
            
            if (self.verbose_mode):
                sys.stderr.write("Iteration %d, average likelihood: %.10f\n"%(iteration, avg_ll))
                
            if (avg_ll - prev_avg_ll < delta):
                break
            
            if (avg_ll < prev_avg_ll):
                sys.stderr.write("Something wrong ... \n")
                return False
            
            prev_avg_ll = avg_ll
            iteration += 1
        
        return True

    # forward_backword algorithm for training
    def forward_backward(self, ob_seq_int):
        N = len(self.st_list)
        ob_seq_len = len(ob_seq_int)

        if (self.verbose_mode):
            sys.stderr.write("\nCompute Xi and Gamma table ... \n")
            
        for t in xrange(ob_seq_len - 1):
            if (self.verbose_mode):
                sys.stderr.write("T %d of %d ... \r"%(t, ob_seq_len))
                
            # compute the denominator for Xi at time t
            Xi_t_denominator = self.__ln(0.0)
            ot_next = ob_seq_int[t + 1]
            for st_from in xrange(N):
                for st_to in xrange(N):
                    Xi_t_denominator = self.__log_add(Xi_t_denominator, self.alpha_table[t + 1][st_from] + self.trans_matrix[st_from][st_to] + self.emit_matrix[st_to][ot_next] + self.beta_table[t + 1][st_to])
                    
            for st_i in xrange(N):
                gamma_t_i = self.__ln(0.0)
                for st_j in xrange(N):
                    Xi_t_numerator = self.alpha_table[t + 1][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot_next] + self.beta_table[t + 1][st_j]
                    Xi_t_i_j = (Xi_t_numerator - Xi_t_denominator)
                    
                    self.xi_table[t][st_i][st_j] = Xi_t_i_j
                    gamma_t_i = self.__log_add(gamma_t_i, Xi_t_i_j)
                self.gamma_table[t][st_i] = gamma_t_i
        
        if (self.verbose_mode):
            sys.stderr.write("\nComputing new values for init_matrix ... \n")
            
        for st_i in xrange(N):
            self.init_matrix_copy[st_i] = self.gamma_table[0][st_i]
        
        if (self.verbose_mode):
            sys.stderr.write("\nComputing new values for trans_matrix ... \n")
            
        for st_i in xrange(N):
            if (self.verbose_mode):
                sys.stderr.write("State %d of %d ... \r"%(st_i, N))
                
            trans_denominator = self.NEG_INF
            
            for t in xrange(ob_seq_len - 1):
                trans_denominator = self.__log_add(trans_denominator, self.gamma_table[t][st_i])
            
            for st_j in xrange(N):
                trans_numerator = self.NEG_INF
                for t in xrange(ob_seq_len - 1):
                    trans_numerator = self.__log_add(trans_numerator, self.xi_table[t][st_i][st_j])
                
                self.trans_matrix_copy[st_i][st_j] = (trans_numerator - trans_denominator)
        
        if (self.verbose_mode):
            sys.stderr.write("\nComputing new values for emit_matrix ... \n")
            
        emit_numerators_list = [self.NEG_INF for vk in xrange(len(self.ob_list))] 
        for st_j in xrange(N):
            if (self.verbose_mode):
                sys.stderr.write("State %d of %d ... \r"%(st_j, N))
                
            emit_denominator = self.NEG_INF
            for vk in xrange(len(self.ob_list)):
                emit_numerators_list[vk] = self.NEG_INF
                
            for t in xrange(ob_seq_len - 1):
                emit_denominator = self.__log_add(emit_denominator, self.gamma_table[t][st_j])
                curr_emit_numerator = emit_numerators_list[ob_seq_int[t]]
                emit_numerators_list[ob_seq_int[t]] = self.__log_add(curr_emit_numerator, self.gamma_table[t][st_j])
            
            for vk in xrange(len(self.ob_list)):
                self.emit_matrix_copy[st_j][vk] = (emit_numerators_list[vk] - emit_denominator)
    
    # computes the new trans[st_i][st_j] based on the __Xi table and gamma table
    def __trans_prime(self, st_i, st_j, ob_seq_int):
        numerator = self.__ln(0.0)
        denominator = self.__ln(0.0)
        ob_seq_len = len(ob_seq_int)

        for t in xrange(ob_seq_len - 1):
            numerator = self.__log_add(numerator, self.xi_table[t][st_i][st_j])
            denominator = self.__log_add(denominator, self.gamma_table[t][st_i])
        
        return (numerator - denominator)
    
    # computes the new emit[st_i][vk] based on __Xi table and gamma table
    def __emit_prime(self, st_j, vk, ob_seq_int):
        numerator = self.__ln(0.0)
        denominator = self.__ln(0.0)
        ob_seq_len = len(ob_seq_int)
        
        for t in xrange(ob_seq_len - 1):
            if (ob_seq_int[t] == vk):
                numerator = self.__log_add(numerator, self.gamma_table[t][st_j])
            denominator = self.__log_add(denominator, self.gamma_table[t][st_j])
        
        return (numerator - denominator)
    
    # add a single state to HMM
    def add_state(self, st):
        self.st_list.append(st)
        self.st_list_index[st] = len(self.st_list) - 1
        
        N = len(self.st_list)
        ob_list_len = len(self.ob_list)
        self.init_matrix.append(self.NEG_INF)
        self.init_matrix_copy.append(self.NEG_INF)
        self.trans_matrix.append([self.NEG_INF for x in xrange(N)])
        self.trans_matrix_copy.append([self.NEG_INF for x in xrange(N)])
        self.emit_matrix.append([self.NEG_INF for x in xrange(ob_list_len)])
        self.emit_matrix_copy.append([self.NEG_INF for x in xrange(ob_list_len)])
    
    # remove a state from HMM
    def remove_state(self, st):
        st_idx = self.st_list_index[st]
        del self.st_list[self.st_list_index[st]]
        del self.st_list_index[st]
        
        del self.init_matrix[st_idx]
        del self.init_matrix_copy[st_idx]
        
        del self.trans_matrix[st_idx]
        for item in self.init_matrix:
            del item[st_idx]
        del self.trans_matrix_copy[st_idx]
        for item in self.init_matrix_copy:
            del item[st_idx]
        
        del self.emit_matrix[st_idx]
        del self.emit_matrix_copy[st_idx]
        
    
    # add an observation to HMM
    def add_observation(self, ob):
        self.ob_list.append(ob)
        self.ob_list_index[ob] = len(self.ob_list) - 1

        N = len(self.st_list)
        for st in xrange(N):
            self.emit_matrix[st].append(self.NEG_INF)
            self.emit_matrix_copy[st].append(self.NEG_INF)            
    
    # remove an observation from HMM
    def remove_observation(self, ob):
        ob_idx = self.ob_list_index[ob]
        del self.ob_list[ob_idx]
        del self.ob_list_index[ob]
        
        N = len(self.st_list)
        for st in xrange(N):
            del self.emit_matrix[st][ob_idx]
            del self.emit_matrix_copy[st][ob_idx]
    
    # set the list of states
    def set_states(self, st_seq):
        self.st_list = copy.copy(st_seq)
        
        # build the index automatically
        self.st_list_index = dict()
        for i in xrange(len(st_seq)):
            self.st_list_index[st_seq[i]] = i
    
    # set the list of observations
    def set_observations(self, ob_seq):
        self.ob_list = copy.copy(ob_seq)
        
        # build the index automatically
        self.ob_list_index = dict()
        for i in xrange(len(ob_seq)):
            self.ob_list_index[ob_seq[i]] = i
    
    # set the initial probability
    def set_initial(self, st, prob):
        self.init_matrix[self.st_list_index[st]] = prob
        
    # set the initial prob matrix, Pi_matrix is a dict
    def set_initial_matrix(self, Pi_matrix):
        ln_0 = self.__ln(0.0)
        N = len(self.st_list)        
        self.init_matrix = [ln_0 for st in xrange(N)]
        self.init_matrix_copy = [ln_0 for st in xrange(N)]
        for st in Pi_matrix:
            self.init_matrix[self.st_list_index[st]] = Pi_matrix[st]
            
    
    # set the transition probability from state[i] to state[j]
    def set_transition(self, st_from, st_to, prob):
        if (st_from not in self.st_list_index):
            self.add_state(st_from)
        
        if (st_to not in self.st_list_index):
            self.add_state(st_to)
        
        self.trans_matrix[self.st_list_index[st_from]][self.st_list_index[st_to]] = prob
    
    # set the transition probability matrix, P(state[i] -> state[j]) = A[i][j]
    def set_transition_matrix(self, A_matrix):
        ln_0 = self.__ln(0.0)
        N = len(self.st_list)
        self.trans_matrix = [[ln_0 for st_i in xrange(N)]  for st_j in xrange(N)]
        self.trans_matrix_copy = [[ln_0 for st_i in xrange(N)]  for st_j in xrange(N)]
        
        for st_i in A_matrix:
            for st_j in A_matrix[st_i]:
                self.trans_matrix[self.st_list_index[st_i]][self.st_list_index[st_j]] = A_matrix[st_i][st_j]
    
    # set the probability of emitting observation[ob] at state[st]
    def set_emission(self, st, ob, prob):
        if (st not in self.st_list_index):
            self.add_state(st)
        
        if (ob not in self.ob_list_index):
            self.add_observation(ob)

        self.emit_matrix[self.st_list_index[st]][self.ob_list_index[ob]] = prob
    
    # set the emission probability matrix
    def set_emission_matrix(self, B_matrix):
        N = len(self.st_list)
        ob_list_len = len(self.ob_list)
        ln_0 = self.__ln(0.0)
        self.emit_matrix = [[ln_0 for st in xrange(ob_list_len)] for ob in xrange(N)]
        self.emit_matrix_copy = [[ln_0 for st in xrange(ob_list_len)] for ob in xrange(N)]
        
        for st in B_matrix:
            for ob in B_matrix[st]:
                self.emit_matrix[self.st_list_index[st]][self.ob_list_index[ob]] = B_matrix[st][ob]
    
    # get the list of states
    def get_states(self):
        return self.st_list
    
    # get the list of valid observations
    def get_observations(self):
        return self.ob_list
    
    # returns the probability of starting from state[st]
    def get_initial(self, st):
        return self.init_matrix[self.st_list_index[st]]
    
    # returns the initial matrix
    def get_initial_matrix(self):
        return self.init_matrix
    
    # returns the transition matrix
    def get_transition_matrix(self):
        return self.trans_matrix
    
    # return the probability of going from state[st_from] to state[st_to]
    def get_transition(self, st_from, st_to):
        return self.trans_matrix[self.st_list_index[st_from]][self.st_list_index[st_to]]
    
    # returns the emission matrix
    def get_emission_matrix(self):
        return self.emit_matrix
    
    # returns the probability of emiting observation[ob] at state[st]
    def get_emission(self, st, ob):
        return self.emit_matrix[self.st_list_index[st]][self.ob_list_index[ob]]
    
    # get a list of random numbers
    def __get_rand_list(self, list_len, sum_to_one = True, take_ln = True):
        result_list = [random.random() for i in xrange(list_len)]
        
        if (sum_to_one):
            list_sum = sum(result_list)
            result_list = map(lambda x: float(x) / float(list_sum), result_list)
        
        if (take_ln):
            result_list = map(lambda x: self.__ln(x), result_list)
        
        return result_list
    
    # Set the verbose mode ON/OFF
    def set_verbose(self, verbose):
        self.verbose_mode = verbose
    
    # randomize the probabilities in init, trans, and emit matrices
    def randomize_matrices(self, seed = None):
        if (seed is not None):
            random.seed(seed)
        else:
            random.seed()
        
        N = len(self.st_list)
        ob_list_len = len(self.ob_list)
        
        self.init_matrix = [0.0 for st in xrange(N)]
        self.init_matrix_copy = [0.0 for st in xrange(N)]
        self.trans_matrix = [[0.0 for st_i in xrange(N)] for st_j in xrange(N)]
        self.trans_matrix_copy = [[0.0 for st_i in xrange(N)] for st_j in xrange(N)]
        self.emit_matrix = [[0.0 for st in xrange(ob_list_len)] for ob in xrange(N)]
        self.emit_matrix_copy = [[0.0 for st in xrange(ob_list_len)] for ob in xrange(N)]
        
        # initial matrix
        rand_list = self.__get_rand_list(N)
        for i in xrange(N):
            self.init_matrix[i] = rand_list[i]
            self.init_matrix_copy[i] = rand_list[i]
        
        # transition matrix
        for i in xrange(N):
            rand_list = self.__get_rand_list(N)
            for j in xrange(N):
                self.trans_matrix[i][j] = rand_list[j]
                self.trans_matrix_copy[i][j] = rand_list[j]
        
        # emission matrix
        for st in xrange(N):
            rand_list = self.__get_rand_list(ob_list_len)
            for ob in xrange(ob_list_len):
                self.emit_matrix[st][ob] = rand_list[ob]
                self.emit_matrix_copy[st][ob] = rand_list[ob]
    
    # load a previously saved model from file
    def read_from_file(self, model_filename):
        fmodel = open(model_filename, "r")
        (states, observations, Pi_matrix, A_matrix, B_matrix) = pickle.load(fmodel)
        fmodel.close()
        
        self.set_states(states)
        self.set_observations(observations)
        self.set_initial_matrix(Pi_matrix)
        self.set_transition_matrix(A_matrix)
        self.set_emission_matrix(B_matrix)
        
    # converts the matrices into dict format for output
    def get_model(self):
        init_matrix_dict = dict()
        trans_matrix_dict = dict()
        emit_matrix_dict = dict()
        
        for st in xrange(len(self.st_list)):
            init_matrix_dict[self.st_list[st]] = self.init_matrix[st]
        
        for st_i in xrange(len(self.st_list)):
            trans_matrix_dict[self.st_list[st_i]] = dict()
            for st_j in xrange(len(self.st_list)):
                trans_matrix_dict[self.st_list[st_i]][self.st_list[st_j]] = self.trans_matrix[st_i][st_j]
        
        for st in xrange(len(self.st_list)):
            emit_matrix_dict[self.st_list[st]] = dict()
            for ob in xrange(len(self.ob_list)):
                emit_matrix_dict[self.st_list[st]][self.ob_list[ob]] = self.emit_matrix[st][ob]
        
        return (self.st_list, self.ob_list, init_matrix_dict, trans_matrix_dict, emit_matrix_dict)
        
    # save the HMM (init, trans, emit) matrices to file
    def write_to_file(self, model_filename):
        fmodel = open(model_filename, 'w')
        pickle.dump(self.get_model(), fmodel)
        fmodel.close()
        
