#!/usr/bin/env python
# lame_tagger.py - A POS tagger based on Hidden Markov Model (using HMMTK)
# Written by Yuchen Zhang
# Dec, 2012

import sys
import pickle
import math
from hmmpytk import hmm_faster

INF = float('inf')    # infinity !!!
NEG_INF = float('-inf')
M_LN2 = 0.69314718055994530942
letters = [chr(x) for x in xrange(ord('a'), ord('z') + 1)]

def ln(value):
    if (value == 0.0):
        return NEG_INF
    else:
        return math.log(value)
    
# calculate log(exp(left) + exp(right)) more accurately
# based on http://www.cs.cmu.edu/~roni/11761-s12/assignments/__log_add.c
def log_add(left, right):
    if (right < left):
        return left + math.log1p(math.exp(right - left))
    elif (right > left):
        return right + math.log1p(math.exp(left - right))
    else:
        return left + M_LN2

# training the hmm using forward-backward algorithm
def train_unsupervised(train_file, model_file):
    ftrain = open(train_file, 'r')
    
    vocab = dict()
    tags = dict()
    all_ob_seqs = list()
    
    # first gathering the vocab and tags
    for line in ftrain:
        line = line.replace("\n", "")
        line = line.replace("\r", "")
        tokens = line.split(" ")
        
        # ob_list = list()
        for t in tokens:
            last_idx = t.rfind("_")
            curr_word = t[:last_idx]
            curr_tag = t[last_idx + 1:]
            
            if (curr_word not in vocab):
                vocab[curr_word] = 1
            else:
                vocab[curr_word] += 1
            
            # ob_list.append(curr_word)
            all_ob_seqs.append(curr_word)
            
            if (curr_tag not in tags):
                tags[curr_tag] = 1
            else:
                tags[curr_tag] += 1

    
    # initialize HMM with tags as states and words as observations, and randomzied probs
    train_hmm = hmm_faster.HMM(states = tags.keys(), observations = vocab.keys())
    train_hmm.randomize_matrices(1123)
    
    # starts training
    train_hmm.train(all_ob_seqs, max_iteration = 1000, delta = 0.001)        
    
    # get prior for each tag
    total_tags = sum([tags[x] for x in tags])
    for t in tags:
        tags[t] = ln(float(tags[t]) / float(total_tags))
    
    fmodel = open(model_file, 'w')
    hmm_model = train_hmm.get_model()
    
    pickle.dump((hmm_model, tags), fmodel)
    fmodel.close()

# assume train_file is tagged following the Stanford POS tagger's convention:
# WORD_TAG
def train_supervised(train_file, model_file):
    ftrain = open(train_file, 'r')
    
    vocab = dict()
    tags = dict()
    tag_bigram = dict()
    word_given_tag = dict()
    
    for line in ftrain:
        line = line.replace("\n", "")
        tokens = line.split(" ")
        prev_tag = "NULL"
        for t in tokens:
            last_idx = t.rfind("_")
            curr_word = t[:last_idx]
            curr_tag = t[last_idx + 1:]
            
            # add to vocab
            vocab[curr_word] = 0
            
            # add to all tags
            if (curr_tag not in tags):
                tags[curr_tag] = 1
            else:
                tags[curr_tag] += 1

            # add to tag_bigram
            if (prev_tag not in tag_bigram):
                tag_bigram[prev_tag] = dict()
            
            if (curr_tag not in tag_bigram[prev_tag]):
                tag_bigram[prev_tag][curr_tag] = 1
            else:
                tag_bigram[prev_tag][curr_tag] += 1
            
            prev_tag = curr_tag
            
            # add to word|tag
            if (curr_tag not in word_given_tag):
                word_given_tag[curr_tag] = dict()
            
            if (curr_word not in word_given_tag[curr_tag]):
                word_given_tag[curr_tag][curr_word] = 1
            else:
                word_given_tag[curr_tag][curr_word] += 1
        
    ftrain.close()
    
    # generate the Pi, A, B matrices
    Pi_matrix = dict()
    A_matrix = dict()
    B_matrix = dict()
    
    # Pi matrix
    sys.stderr.write("Computing the Pi matrix ... \n")
    total_sum = sum([tag_bigram['NULL'][w] for w in tag_bigram['NULL'].keys()])
    for tag in tags:
        sys.stderr.write("%s "%(tag))
        if (tag not in tag_bigram['NULL']):
            count = 1
            total_sum += 1
        else:
            count = tag_bigram['NULL'][tag]
            
        prob = float(count) / float(total_sum)        
        Pi_matrix[tag] = ln(prob)
    
    # A matrix
    sys.stderr.write("\n\nComputing the A matrix ... \n")
    for prev_tag in tags:
        sys.stderr.write("%s "%(prev_tag))
        total_sum = sum([tag_bigram[prev_tag][w] for w in tag_bigram[prev_tag]])
        if (prev_tag not in A_matrix):
            A_matrix[prev_tag] = dict()
            
        for curr_tag in tags:
            if (curr_tag not in tag_bigram[prev_tag]):
                count = 1
                total_sum += 1
            else:
                count = tag_bigram[prev_tag][curr_tag]
                
            prob = float(count) / float(total_sum)
            
            A_matrix[prev_tag][curr_tag] = ln(prob)
            
    # B matrix
    sys.stderr.write("\n\nComputing the B matrix ... \n")
    for tag in tags:
        sys.stderr.write("%s "%(tag))
        total_sum = sum([word_given_tag[tag][w] for w in word_given_tag[tag]])
        if (tag not in B_matrix):
            B_matrix[tag] = dict()
        
        for word in vocab:
            if (word not in word_given_tag[tag]):
                count = 1
                total_sum += 1
            else:
                count = word_given_tag[tag][word]
                
            prob = float(count) / float(total_sum)
            B_matrix[tag][word] = ln(prob)
    
    sys.stderr.write("\n\n")
    
    # get prior for each tag
    total_tags = sum([tags[x] for x in tags])
    for t in tags:
        tags[t] = ln(float(tags[t]) / float(total_tags))
    
    fmodel = open(model_file, 'w')
    hmm_model = (tags.keys(), vocab.keys(), Pi_matrix, A_matrix, B_matrix)
    
    pickle.dump((hmm_model, tags), fmodel)
    fmodel.close()

def tag_string(target_str, hmm_model, prior_dict):
    target_str = target_str.lower()
    st = 0  # 0: letters, 1: punc
    curr_idx = 0
    curr_start = 0
    tokens = list()
    
    while (curr_idx < len(target_str)):
        if (st == 0):
            if (target_str[curr_idx] in letters):
                curr_idx += 1
            else:
                if (curr_idx > curr_start):
                    tokens.append(target_str[curr_start:curr_idx])
                curr_start = curr_idx
                if (target_str[curr_idx] == ' '):
                    st = 2
                else:
                    st = 1
                    
        elif (st == 1):
            stay = False
            if (curr_idx == curr_start):
                if (target_str[curr_idx] not in letters and target_str[curr_idx] != ' '):
                    stay = True
            else:
                if (target_str[curr_idx] == target_str[curr_idx - 1]):
                    stay = True
            
            if (stay):
                curr_idx += 1
            else:
                if (curr_idx > curr_start):
                    tokens.append(target_str[curr_start:curr_idx])
                    curr_start = curr_idx
                    if (target_str[curr_idx] == ' '):
                        st = 2
                    else:
                        st = 0

        elif (st == 2):
            if (target_str[curr_idx] == ' '):
                curr_idx += 1
            else:
                curr_start = curr_idx
                if (target_str[curr_idx] in letters):
                    st = 0
                else:
                    st = 1
    if (curr_idx > curr_start):
        tokens.append(target_str[curr_start:curr_idx])
    
    tokens = filter(lambda x:len(x) > 0, tokens)
    
    # use prior distribution of each tag to estimate 
    # P(unknown word | tag) = Prior[tag] ^ 2
    for w in tokens:
        if (w not in hmm_model.get_observations()):
            for tag in prior_dict:
                hmm_model.set_emission(tag, w, prior_dict[tag] * 2)
    
    # normalize the emission matrix for the hmm
    emit_matrix = hmm_model.get_emission_matrix()
    for st in xrange(len(emit_matrix)):
        log_sum = NEG_INF
        for ob in xrange(len(emit_matrix[st])):
            log_sum = log_add(log_sum, emit_matrix[st][ob])
        
        for ob in xrange(len(emit_matrix[st])):
            emit_matrix[st][ob] = emit_matrix[st][ob] - log_sum
    
    # viterbi
    tag_seq = hmm_model.viterbi(tokens)
    # print tag_seq
    return (tokens, tag_seq)

def tag_file(target_file, hmm_model, prior):
    ftarget = open(target_file)
    for line in ftarget:
        line = line.replace("\n", "")
        line = line.replace("\r", "")
        (tokens, tag_seq) = tag_string(line, hmm_model, prior)
        
        if (tag_seq is not None):
            for i in xrange(0, len(tokens)):
                sys.stdout.write("%s_%s "%(tokens[i], tag_seq[i]))
                sys.stderr.write("%s_%s "%(tokens[i], tag_seq[i]))
        sys.stdout.write("\n")
        sys.stderr.write("\n")
        
    ftarget.close()

def print_usage():
    sys.stderr.write("Training the POS tagger (supervised and unsupervised):\n")
    sys.stderr.write("./lame_tagger.py --train-supervised train-file.txt model-file.hmm\n")
    sys.stderr.write("./lame_tagger.py --train-unsupervised train-file.txt model-file.hmm\n")
    sys.stderr.write("Tag a file:\n")
    sys.stderr.write("./lame_tagger.py --tag test-file.txt model-file.hmm\n")

def main():
    if (len(sys.argv) == 1):
        print_usage()
        return 0
    
    idx = 1
    while (idx < len(sys.argv)):
        curr_argv = sys.argv[idx].lower()
        if (curr_argv == '--train-supervised' or curr_argv == '--train-unsupervised'):
            if (idx + 2 >= len(sys.argv)):
                sys.stderr.write("Must specify train_manual file and model file.\n")
                print_usage()
                return 2
            
            train_file = sys.argv[idx + 1]
            model_file = sys.argv[idx + 2]
            
            if (curr_argv == "--train-supervised"):
                train_supervised(train_file, model_file)
            elif (curr_argv == '--train-unsupervised'):
                train_unsupervised(train_file, model_file)
            break
        
        elif (curr_argv == '--tag'):
            if (idx + 2 >= len(sys.argv)):
                sys.stderr.write("Must specify test file and model file.\n")
                print_usage()
                return 3
            test_file = sys.argv[idx + 1]
            model_file = sys.argv[idx + 2]
            
            fmodel = open(model_file)
            (hmm_matrices, prior) = pickle.load(fmodel)
            fmodel.close()
            
            (st_list, ob_list, Pi, A, B) = hmm_matrices
            hmm_model = hmm_faster.HMM(st_list, ob_list, Pi, A, B)
            
            tag_file(test_file, hmm_model, prior)
            break
            
        else:
            sys.stderr.write("Invalid argument %s\n"%(sys.argv[idx]))
            print_usage()
            break
    

if __name__ == "__main__":
    sys.exit(main())
    