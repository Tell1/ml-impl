#!/usr/bin/env python
import sys
from array import array
import random
import time
import hmm_faster
import math

M_LN2 = 0.69314718055994530942

def log_add(left, right):
    if (right < left):
        return left + math.log1p(math.exp(right - left))
    elif (right > left):
        return right + math.log1p(math.exp(left - right))
    else:
        return left + M_LN2


def log_add2(left, right):
    # if (right < left):
        # return left + math.log1p(math.exp(right - left))
    return right + math.log1p(math.exp(left - right))


def main():
    
    N = 4000000
    ans = 0.0
    
    print log_add(-3.4, -3.4)
    print log_add2(-3.4, -3.4)
    
    
    start_time = time.time()
    for i in xrange(N):
        rnd1 = -random.random()
        rnd2 = -random.random()
        ans = log_add2(rnd1, rnd2)
    end_time = time.time()
    print (end_time - start_time)
    
    start_time = time.time()
    for i in xrange(N):
        rnd1 = -random.random()
        rnd2 = -random.random()
        # ans = max(rnd1, rnd2) + math.log1p(math.exp(-abs(rnd1 - rnd2)))
        # ans = log_add2(rnd1, rnd2)
        # g = lambda left, right: left + right 
            # if (right < left): left + math.log1p(math.exp(right - left) else: right + math.log1p(math.exp(left - right)))
        # ans =  (rnd1, rnd2)
        ans = rnd1 + math.log1p(math.exp(rnd1 - rnd2))
    end_time = time.time()
    print (end_time - start_time)
    
    
    
    sys.exit(0)
    
    hmm_train = hmm_faster.HMM()
    hmm_train.set_states([0, 1])
    obs = [chr(x) for x in xrange(ord('A'), ord('Z') + 1)]
    obs.append(" ")
    hmm_train.set_observations(obs)
    hmm_train.randomize_matrices(1123)
    
    finput = open("hmm-train.clean.txt", 'r')
    for line in finput:
        line = line.replace("\n", "")
        hmm_train.train(line[:1000])
        
    finput.close()
    
    sys.exit(0)
    
    N = 100000000
    li = [0 for i in xrange(N)]
        
    start_time = time.time()
    a = 0
    for i in xrange(N):
        a = i
    end_time = time.time()
    print (end_time - start_time)
    
    start_time = time.time()
    a = 0
    idx = 0
    for i in xrange(len(li)):
        a = i
    end_time = time.time()
    print (end_time - start_time)
    
    sys.exit(0)
    
    mylist = list([0] * N)
    # myarr = array('f', mylist)
    mydict = dict()
    for i in xrange(0, N):
        mydict[(i,i * 2, "p")] = 0
    
    start_time = time.time()
    for i in xrange(0, N):
        rnd = random.random()
        # myarr.append(rnd)
        mydict[(i,i * 2, "p")] = rnd        
    end_time = time.time()
    print (end_time - start_time)

    start_time = time.time()
    for i in xrange(0, N):
        rnd = random.random()
        # myarr.append(rnd)
        mylist[i] = rnd
    end_time = time.time()
    print (end_time - start_time)
    
    sys.exit(0)
    
    start_time = time.time()
    for i in xrange(0, 10000000):
        rnd = random.random()
        myarr.append(rnd)        
    end_time = time.time()
    print (end_time - start_time)
    
    start_time = time.time()
    for i in xrange(0, 10000000):
        rnd = random.random()
        mylist.append(rnd)        
    end_time = time.time()
    print (end_time - start_time)
    

if __name__ == "__main__":
    main()