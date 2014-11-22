# hmmtk/hmm.py - HMM (Hidden Markov Model) implementation written in Python 
# First version author: Yuchen Zhang (yuchenz@cs.cmu.edu)
# Version history:
# 
# Dec 28, 2012, 0.1.1 - Added support for training on multiple instances
# Dec 27, 2012, 0.1.0 - initial version
# 
# *** THIS PACKAGE IS OBSOLETE, PLEASE USE THE MORE CURRENT hmm_faster.py *** 
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
        self.st_list = None
        self.ob_list = None
        self.init_matrix = None
        self.trans_matrix = None
        self.emit_matrix = None
        self.alpha_table = None
        self.beta_table = None
        
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
        
        viterbi_table = list()
        bp_table = list()
        viterbi_table.append(copy.copy(self.init_matrix))
        
        t = 1
        while (t <= len(ob_seq)): # loop through time
            viterbi_t = dict()
            bp_t = dict()
            ot = ob_seq[t - 1]   # this is t - 1 because t0 is start state
            j = 0
            while (j < N): # for each state
                st_j = self.st_list[j]
                i = 0
                viterbi_t_j = self.NEG_INF
                curr_best_st = None
                  
                while (i < N): # calculate alpha_t_j
                    st_i = self.st_list[i]
                    log_sum = viterbi_table[t - 1][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot]

                    if (log_sum > viterbi_t_j):
                        viterbi_t_j = log_sum
                        curr_best_st = st_i
                    i += 1
#                Slow, do not use. Premature optimization is the root of all evil. 
#                (curr_best_st, viterbi_t_j) = max(map(lambda st_i:(st_i, viterbi_table[t - 1][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot]), self.st_list), key=lambda x:x[1])
              
                viterbi_t[st_j] = viterbi_t_j
                bp_t[st_j] = curr_best_st
                j += 1
            viterbi_table.append(viterbi_t)
            bp_table.append(bp_t)
            t += 1
        
        # perform the viterbi back-trace
        t = len(bp_table) - 1
        if (t < 0):
            return None
        states_seq = list()
        
        curr_state = self.st_list[0]
        for st in self.st_list:
            if (bp_table[-1][st] > bp_table[-1][curr_state]):
                curr_state = st
        
        while (t >= 0):
            if (curr_state is None):
                return None
            states_seq.append(curr_state)
            curr_state = bp_table[t][curr_state]
            t -= 1
        
        states_seq.reverse()
        return states_seq
    
    # given the observation sequence, return its probability given the model
    def forward(self, ob_seq):
        N = len(self.st_list)
        
        alpha_table = list()
        alpha_table.append(copy.copy(self.init_matrix))
        
        t = 1
        while (t <= len(ob_seq)): # loop through time
            # sys.stderr.write("\rComputing alpha table t = %d out of %d"%(t, len(ob_seq)))            
            alpha_t = dict()
            ot = ob_seq[t - 1]   # this is t - 1 because t0 is start state
            j = 0
            while (j < N): # for each state
                st_j = self.st_list[j]
                i = 0
                alpha_t_j = self.NEG_INF
                while (i < N):
                    st_i = self.st_list[i]
                    log_sum = alpha_table[t - 1][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot]
                    alpha_t_j = self.__log_add(alpha_t_j, log_sum)
                    i += 1
                    
                alpha_t[st_j] = alpha_t_j
                j += 1
            alpha_table.append(alpha_t)
            t += 1
        
        return alpha_table[1:]

    
    # backward algorithm
    def backward(self, ob_seq):
        N = len(self.st_list)
        
        beta_table = list()
        
        end_entry = dict()
        # follows the paper, not the book
        for st in self.st_list:
            end_entry[st] = self.__ln(1.0)
    
        # create the beta table in advance since we are filling backwards
        t = 0
        while (t < len(ob_seq)):
            beta_t = dict()
            j = 0
            while (j < N):
                st_j = self.st_list[j]
                beta_t[st_j] = self.__ln(0.0)
                j += 1
            
            beta_table.append(beta_t)
            t += 1
        
        beta_table.append(end_entry)
        
        t = len(ob_seq) - 1 # starting from 2nd to last and move backwards
        while (t >= 0):
            # sys.stderr.write("\rComputing beta table t = %d out of %d"%(t, len(ob_seq)))            
            ot_next = ob_seq[t]
            i = 0
            while (i < N):
                st_i = self.st_list[i]
                j = 0
                beta_t_i = self.NEG_INF
                while (j < N):
                    st_j = self.st_list[j]
                    log_sum = self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot_next] + beta_table[t + 1][st_j]
                    beta_t_i = self.__log_add(beta_t_i, log_sum)
                    j += 1
    
                beta_table[t][st_i] = beta_t_i
                i += 1
                
            t -= 1
        
        return beta_table[:-1]

    # computes the __Xi[t][i][j]: being at st[i] at time t, and at st[j] at time t + 1
    def __Xi(self, st_i, st_j, t, ob_seq, alpha, beta):
        ot_next = ob_seq[t + 1]
        
        numerator = alpha[t][st_i] + self.trans_matrix[st_i][st_j] + self.emit_matrix[st_j][ot_next] + beta[t + 1][st_j]
        denominator = self.__ln(0.0)
        
        for st_from in self.st_list:
            for st_to in self.st_list:
                denominator = self.__log_add(denominator, alpha[t][st_from] + self.trans_matrix[st_from][st_to] + self.emit_matrix[st_to][ot_next] + beta[t + 1][st_to])

        return (numerator - denominator) 

    # get average likelihood from alpha table
    def __get_avgll_by_alpha(self, alpha_table, ob_seq):
        log_sum = self.NEG_INF
        for st in self.st_list:
            log_sum = self.__log_add(log_sum, alpha_table[-1][st])
        
        return (log_sum / float(len(ob_seq)))

    # train the HMM using forward-backward algorithm, stops when the differences 
    # between likelihoods is smaller than delta, or reaches maximum iteration
    def train_multiple(self, ob_seq_list, max_iteration = 100, delta = 0.000001):
        if (max_iteration < 1):
            return None

        iteration = 0
        prev_avg_ll = self.NEG_INF
        
        while (iteration < max_iteration):
            alpha_table_list = list()
            beta_table_list = list()
            ob_seq_num = len(ob_seq_list)
            
            avg_ll = self.NEG_INF
            for k in xrange(0, ob_seq_num):
                sys.stderr.write("\rComputing Alpha and Beta tables for observation %d of %d ... "%(k, ob_seq_num))
                ob_seq = ob_seq_list[k]
                alpha_table = self.forward(ob_seq)
                beta_table = self.backward(ob_seq)
                
                alpha_table_list.append(alpha_table)
                beta_table_list.append(beta_table)
                
                avg_ll = self.__log_add(avg_ll, self.__get_avgll_by_alpha(alpha_table, ob_seq))
            avg_ll -= self.__ln(ob_seq_num)
            
            (init_matrix_new, trans_matrix_new, emit_matrix_new) = self.forward_backward_multiple(ob_seq_list, alpha_table_list, beta_table_list)
            self.init_matrix = init_matrix_new
            self.trans_matrix = trans_matrix_new
            self.emit_matrix = emit_matrix_new
            
            sys.stderr.write("Iteration %d, average likelihood: %.10f\n"%(iteration, avg_ll))
            if (avg_ll - prev_avg_ll < delta):
                break
            
            if (avg_ll < prev_avg_ll):
                sys.stderr.write("Something wrong ... \n")
                return False
            
            prev_avg_ll = avg_ll            
            iteration += 1
        
        return True

    # train the HMM using forward-backward algorithm, stops when the differences 
    # between likelihoods is smaller than delta, or reaches maximum iteration
    def train(self, ob_seq, max_iteration = 100, delta = 0.000001):
        if (max_iteration < 1):
            return None

        iteration = 0
        prev_avg_ll = self.NEG_INF
        
        while (iteration < max_iteration):
            alpha_table = self.forward(ob_seq)
            beta_table = self.backward(ob_seq)
            avg_ll = self.__get_avgll_by_alpha(alpha_table, ob_seq)
            
            (init_matrix_new, trans_matrix_new, emit_matrix_new) = self.forward_backward(ob_seq, alpha_table, beta_table)
            self.init_matrix = init_matrix_new
            self.trans_matrix = trans_matrix_new
            self.emit_matrix = emit_matrix_new
            
            sys.stderr.write("Iteration %d, average likelihood: %.10f\n"%(iteration, avg_ll))
            if (avg_ll - prev_avg_ll < delta):
                break
            
            if (avg_ll < prev_avg_ll):
                sys.stderr.write("Something wrong ... \n")
                return False
            
            prev_avg_ll = avg_ll            
            iteration += 1
        
        return True

    # forward-backward algorithm with multiple training instances, following 
    # the assumption that each training example are statistically independent
    def forward_backward_multiple(self, ob_seq_list, alpha_table_list, beta_table_list):
        trans_matrix_new = dict()
        emit_matrix_new = dict()
        init_matrix_new = dict()
        N = len(self.st_list)
        
        Xi_table_list = list()
        gamma_table_list = list()
        
        k = 0
        while (k < len(ob_seq_list)):
            sys.stderr.write("\rComputing Xi and gamma tables for observation %d of %d ..."%(k, len(ob_seq_list)))
            ob_seq = ob_seq_list[k]
            alpha_table = alpha_table_list[k]
            beta_table = beta_table_list[k]
            # first compute Xi_table[t][i][j] and gamma_table[t][i]
            # sys.stderr.write("Calculating __Xi, gamma tables ... \n")
            Xi_table = list()
            gamma_table = list()
            t = 0
            while (t < len(ob_seq) - 1):
                i = 0
                Xi_t = dict()
                gamma_t = dict()
                while (i < N):
                    st_i = self.st_list[i]
                    Xi_t_i = dict()
                    gamma_t_i = self.__ln(0.0)
                    j = 0
                    
                    while (j < N):
                        st_j = self.st_list[j]
                        Xi_t_i_j = self.__Xi(st_i, st_j, t, ob_seq, alpha_table, beta_table)
                        Xi_t_i[st_j] = Xi_t_i_j
                        gamma_t_i = self.__log_add(gamma_t_i, Xi_t_i_j)
                        j += 1
                        
                    Xi_t[st_i] = Xi_t_i
                    gamma_t[st_i] = gamma_t_i
                    i += 1
                    
                Xi_table.append(Xi_t)
                gamma_table.append(gamma_t)
                t += 1
            Xi_table_list.append(Xi_table)
            gamma_table_list.append(gamma_table)
            k += 1
        
        # sys.stderr.write("Computing new values for init_matrix ... \n")
        for st_i in self.st_list:
            init_st_i = self.NEG_INF
            ob_seq_num = len(ob_seq_list)
            for k in xrange(0, ob_seq_num):
                init_st_i = self.__log_add(init_st_i, gamma_table_list[k][0][st_i])
                # init_matrix_new[st_i] = gamma_table[0][st_i]
            init_st_i -= self.__ln(ob_seq_num)
            init_matrix_new[st_i] = init_st_i
        
        # sys.stderr.write("Computing new values for trans_matrix ... \n")
        for st_i in self.st_list:
            trans_matrix_new[st_i] = dict()
            for st_j in self.st_list:
                trans_matrix_new[st_i][st_j] = self.__trans_prime_multiple(st_i, st_j, Xi_table_list, gamma_table_list, ob_seq_list)
        
        # sys.stderr.write("Computing new values for emit_matrix ... \n")
        for st_j in self.st_list:
            emit_matrix_new[st_j] = dict()
            for vk in self.ob_list:
                emit_matrix_new[st_j][vk] = self.__emit_prime_multiple(st_j, vk, Xi_table_list, gamma_table_list, ob_seq_list)

        return (init_matrix_new, trans_matrix_new, emit_matrix_new)

    # forward_backword algorithm for training
    def forward_backward(self, ob_seq, alpha_table, beta_table):
        trans_matrix_new = dict()
        emit_matrix_new = dict()
        init_matrix_new = dict()
        N = len(self.st_list)
        
        # first compute Xi_table[t][i][j] and gamma_table[t][i]
        # sys.stderr.write("Calculating __Xi, gamma tables ... \n")
        Xi_table = []
        gamma_table = []
        t = 0
        while (t < len(ob_seq) - 1):
            i = 0
            Xi_t = dict()
            gamma_t = dict()
            while (i < N):
                st_i = self.st_list[i]
                Xi_t_i = dict()
                gamma_t_i = self.__ln(0.0)
                j = 0
                
                while (j < N):
                    st_j = self.st_list[j]
                    Xi_t_i_j = self.__Xi(st_i, st_j, t, ob_seq, alpha_table, beta_table)
                    Xi_t_i[st_j] = Xi_t_i_j
                    gamma_t_i = self.__log_add(gamma_t_i, Xi_t_i_j)
                    j += 1
                    
                Xi_t[st_i] = Xi_t_i
                gamma_t[st_i] = gamma_t_i
                i += 1
                
            Xi_table.append(Xi_t)
            gamma_table.append(gamma_t)
            t += 1
        
        # sys.stderr.write("Computing new values for init_matrix ... \n")
        for st_i in self.st_list:
            init_matrix_new[st_i] = gamma_table[0][st_i]
        
        # sys.stderr.write("Computing new values for trans_matrix ... \n")
        for st_i in self.st_list:
            trans_matrix_new[st_i] = dict()
            for st_j in self.st_list:
                trans_matrix_new[st_i][st_j] = self.__trans_prime(st_i, st_j, Xi_table, gamma_table, ob_seq)
        
        # sys.stderr.write("Computing new values for emit_matrix ... \n")
        for st_j in self.st_list:
            emit_matrix_new[st_j] = dict()
            for vk in self.ob_list:
                emit_matrix_new[st_j][vk] = self.__emit_prime(st_j, vk, Xi_table, gamma_table, ob_seq)

        return (init_matrix_new, trans_matrix_new, emit_matrix_new)

    # computes the new trans[st_i][st_j] based on the __Xi table and gamma table
    def __trans_prime_multiple(self, st_i, st_j, Xi_table_list, gamma_table_list, ob_seq_list):
        numerator = self.__ln(0.0)
        denominator = self.__ln(0.0)
        
        for k in xrange(0, len(ob_seq_list)):
            Xi_table = Xi_table_list[k]
            gamma_table = gamma_table_list[k]
            ob_seq = ob_seq_list[k]
            t = 0
            while (t < len(ob_seq) - 1):
                numerator = self.__log_add(numerator, Xi_table[t][st_i][st_j])
                denominator = self.__log_add(denominator, gamma_table[t][st_i])
                t += 1
        
        return (numerator - denominator)
    
    # computes the new emit[st_i][vk] based on __Xi table and gamma table
    def __emit_prime_multiple(self, st_j, vk, Xi_table_list, gamma_table_list, ob_seq_list):
        numerator = self.__ln(0.0)
        denominator = self.__ln(0.0)
        
        for k in xrange(0, len(ob_seq_list)):
            gamma_table = gamma_table_list[k]
            ob_seq = ob_seq_list[k]
            t = 0
            while (t < len(ob_seq) - 1):
                if (ob_seq[t] == vk):
                    numerator = self.__log_add(numerator, gamma_table[t][st_j])
                denominator = self.__log_add(denominator, gamma_table[t][st_j])
                t += 1
        
        return (numerator - denominator)
    
    # computes the new trans[st_i][st_j] based on the __Xi table and gamma table
    def __trans_prime(self, st_i, st_j, Xi_table, gamma_table, ob_seq):
        numerator = self.__ln(0.0)
        denominator = self.__ln(0.0)
        
        t = 0
        while (t < len(ob_seq) - 1):
            numerator = self.__log_add(numerator, Xi_table[t][st_i][st_j])
            denominator = self.__log_add(denominator, gamma_table[t][st_i])
            t += 1
        
        return (numerator - denominator)
    
    # computes the new emit[st_i][vk] based on __Xi table and gamma table
    def __emit_prime(self, st_j, vk, Xi_table, gamma_table, ob_seq):
        numerator = self.__ln(0.0)
        denominator = self.__ln(0.0)
        
        t = 0
        while (t < len(ob_seq) - 1):
            if (ob_seq[t] == vk):
                numerator = self.__log_add(numerator, gamma_table[t][st_j])
            denominator = self.__log_add(denominator, gamma_table[t][st_j])
            t += 1
        
        return (numerator - denominator)
    
    # set the list of states
    def set_states(self, st_seq):
        self.st_list = copy.copy(st_seq)
    
    # set the list of observations
    def set_observations(self, ob_seq):
        self.ob_list = copy.copy(ob_seq)
    
    # set the initial probability
    def set_initial(self, st, prob):
        self.init_matrix[st] = prob
        
    # set the initial prob matrix
    def set_initial_matrix(self, Pi_matrix):
        self.init_matrix = copy.copy(Pi_matrix)
        for st in self.st_list:
            if (st not in self.init_matrix):
                self.init_matrix[st] = self.__ln(0.0)
    
    # set the transition probability from state[i] to state[j]
    def set_transition(self, st_from, st_to, prob):
        self.trans_matrix[st_from][st_to] = prob
    
    # set the transition probability matrix, P(state[i] -> state[j]) = A[i][j]
    def set_transition_matrix(self, A_matrix):
        self.trans_matrix = copy.copy(A_matrix)
        for from_st in self.st_list:
            if (from_st not in self.trans_matrix):
                self.trans_matrix[from_st] = dict()
            for to_st in self.st_list:
                if (to_st not in self.trans_matrix[from_st]):
                    self.trans_matrix[from_st][to_st] = self.__ln(0.0)           
    
    # set the probability of emitting observation[ob] at state[st]
    def set_emission(self, st, ob, prob):
        self.emit_matrix[st][ob] = prob
    
    # set the emission probability matrix
    def set_emission_matrix(self, B_matrix):
        self.emit_matrix = copy.copy(B_matrix)
        for st in self.st_list:
            if (st not in self.emit_matrix):
                self.emit_matrix[st] = dict()
            for ob in self.ob_list:
                if (ob not in self.emit_matrix[st]):
                    self.emit_matrix[st][ob] = self.__ln(0.0)
    
    # get the list of states
    def get_states(self):
        return self.st_list
    
    # get the list of valid observations
    def get_observations(self):
        return self.ob_list
    
    # returns the probability of starting from state[st]
    def get_initial(self, st):
        return self.init_matrix[st]
    
    # returns the initial matrix
    def get_initial_matrix(self):
        return self.init_matrix
    
    # returns the transition matrix
    def get_transition_matrix(self):
        return self.trans_matrix
    
    # return the probability of going from state[st_from] to state[st_to]
    def get_transition(self, st_from, st_to):
        return self.trans_matrix[st_from][st_to]
    
    # returns the emission matrix
    def get_emission_matrix(self):
        return self.emit_matrix
    
    # returns the probability of emiting observation[ob] at state[st]
    def get_emission(self, st, ob):
        return self.emit_matrix[st][ob]
    
    # get a list of random numbers
    def __get_rand_list(self, list_len, sum_to_one = True, take_ln = True):
        result_list = [random.random() for i in xrange(0, list_len)]
        
        if (sum_to_one):
            list_sum = sum(result_list)
            result_list = map(lambda x: float(x) / float(list_sum), result_list)
        
        if (take_ln):
            result_list = map(lambda x: self.__ln(x), result_list)
        
        return result_list
    
    # randomize the probabilities in init, trans, and emit matrices
    def randomize_matrices(self, seed = None):
        if (seed is not None):
            random.seed(seed)
        else:
            random.seed()
        
        self.init_matrix = dict()
        self.trans_matrix = dict()
        self.emit_matrix = dict()
        N = len(self.st_list)
        
        # initial matrix
        rand_list = self.__get_rand_list(N)
        for i in xrange(0, N):
            st = self.st_list[i]
            self.init_matrix[st] = rand_list[i]
        
        # transition matrix
        for i in xrange(0, N):
            st_i = self.st_list[i]
            self.trans_matrix[st_i] = dict()
            rand_list = self.__get_rand_list(N)
            for j in xrange(0, N):
                st_j = self.st_list[j]
                self.trans_matrix[st_i][st_j] = rand_list[j]
        
        # emission matrix
        for st in self.st_list:
            rand_list = self.__get_rand_list(len(self.ob_list))
            self.emit_matrix[st] = dict()
            for i in xrange(0, len(self.ob_list)):
                ob = self.ob_list[i]
                self.emit_matrix[st][ob] = rand_list[i]
    
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
        
    # save the HMM (init, trans, emit) matrices to file
    def write_to_file(self, model_filename):
        fmodel = open(model_filename, 'w')
        pickle.dump((self.st_list, self.ob_list, self.init_matrix, self.trans_matrix, self.emit_matrix), fmodel)
        fmodel.close()
        
