# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 23:57:15 2017

@author: Shiro
"""
import numpy as np
import time
class R2_EMOAs():
    def __init__(self):
       pass
   
    def get_weights_uniform_2m(self, k):
        return np.asarray([[(1/(k-1))*i, 1-(1/(k-1))*i] for i in range(0,k)])

    def r2_indicator(self, A, weights, utopian):
        #[A_n, A_m] = A.shape
        #[weights_n, wieghts_m] = weights.shape
        begin = time.time()
        delta = weights.shape
        
        r_final = []
        for l in weights: # lambda
            r_int = []
            for a in A: # point
                r_int.append( max( np.multiply(l, abs(utopian-a) ) ) )
            r_final.append(min(r_int))
        r_final = np.asarray(r_final)
        r2_indicator = r_final.sum()/delta[0]
        print('temps de calcul :', time.time()-begin)

        return r2_indicator 
        
    def r2_indicator_optimize(self, A, weights, utopian):
        #[A_n, A_m] = A.shape
        #[weights_n, wieghts_m] = weights.shape
        begin = time.time()
        delta = weights.shape
        abs_points = abs(utopian-A)
        r_final = []
        for l in weights:
            R2_intermediaire = np.multiply(l, abs_points)
            r_final.append(min(np.amax(R2_intermediaire, axis = 1)))
        r_final = np.asarray(r_final)
        r2_indicator = r_final.sum()/delta[0]
        print('temps de calcul :', time.time()-begin)
        return r2_indicator 
    
    def argmin_ra(self, Rh, weights, utopian):
        ra = []
        for index, a in enumerate(Rh):
            Rh_minus_a = np.delete(Rh, index, axis=0) # remove a row depending of the index
            ra.append(self.r2_indicator(Rh_minus_a, weights, utopian)) # compute r2_indicator and save it
        
        i = ra.index(min(ra)) # minimum      
        return Rh[i], i
                
## Test        



l = np.asarray([[0, 1],[1,0],[0.5, 0.5]])
utopian = np.asarray([-0.1, -0.1])
print(l)
a = np.asarray([[10, -5],[7, 9], [1, 2]])

R2= R2_EMOAs()
weights = R2.get_weights_uniform_2m(3)
print(R2.r2_indicator(a, weights,utopian)) # 4.25/3 true resultat
print(l.shape)

print(np.amax(l, axis=0))
print(R2.r2_indicator_optimize(a, weights,utopian))
print(R2.argmin_ra(a, weights, utopian))