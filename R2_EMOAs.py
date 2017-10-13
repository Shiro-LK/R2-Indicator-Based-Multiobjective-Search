# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 23:57:15 2017
@author: Shiro
@author: Vincent Boyer
"""

from __future__ import division
import numpy as np
import time
import math
import random

# Global variable - Constant

VALUE_INIT = 0
IGNORE_OBJECTIVE_VALUE_INDEX = 4
NO_NORMALIZED_VALUE_LOW_INDEX = -4
NO_NORMALIZED_VALUE_HIGH_INDEX = -2
NORMALIZED_VALUE_LOW_INDEX = -2
OBJECTIF_1 = -4
OBJECTIF_2 = -3

class R2_EMOAs():

    def __init__(self, fun):
        """
        Initialization of a R2-EMOA object according to fun function set

        :param fun: functions set from Suite object (class defined in coco library)
        """
        self.population_list = []
        print("Initializing R2_EMOA object")

    def normalize_points(self, R):
        """
        Normalization of values in the objective plan and Rh

        :param R: Pareto front
        :return: worst pareto front with normalized value
        """

        Rh = []
        f1_max = max(self.population_list[:,OBJECTIF_1])
        f2_max = max(self.population_list[:,OBJECTIF_2])
        f1_min = min(self.population_list[:,OBJECTIF_1])
        f2_min = min(self.population_list[:,OBJECTIF_2])

        self.population_list[:,NORMALIZED_VALUE_LOW_INDEX:] = self.population_list[:,NO_NORMALIZED_VALUE_LOW_INDEX:NO_NORMALIZED_VALUE_HIGH_INDEX] - np.array([f1_min, f2_min])
        self.population_list[:,NORMALIZED_VALUE_LOW_INDEX:] = self.population_list[:,NORMALIZED_VALUE_LOW_INDEX:] / np.array([f1_max - f1_min, f2_max - f2_min])

        R_worst_normalize = (R[len(R)-1] - np.array([f1_min, f2_min]))/ np.array([f1_max - f1_min, f2_max - f2_min])

        for tab in R_worst_normalize:
            Rh.append(tab)

        return Rh

    def get_weights_uniform_2m(self, k):
        """
        Compute uniforme weighting vector (2 dimensions)
        :param k: number of vectors
        :return: vector list - numpy array (k,2)
        """

        return np.asarray([[(1 / (k - 1)) * i, 1 - (1 / (k - 1)) * i] for i in range(0, k)])

    def r2_indicator_optimize(self, point, weights, utopian):
        """
        Compute R2 indicator from a utopian point, weigthing factor and values

        R2(A,Λ, vect(r∗)) = (1/Λ) * sum(λ ∈ Λ){ min(vect(a) ∈ A){ max(j ∈ {1,...,m}){ λj * |vect(r∗)j − vect(a)j| }}

        :param point: array N * 2 - set of 2D-points
        :param weights: array M * 2 - set of 2D-weigthing factor
        :param utopian: 2D-points
        :return: R2 value
        """

        return np.array([min(np.amax(np.multiply(wgts, abs(utopian - point)), axis=1)) for wgts in weights]).sum() / weights.shape[0]

    def argmin_ra(self, Rh, weights, utopian):
        """
        ∀ vect(a) ∈ Rh :r(vect(a)) = R2( Rh \ { vect(a)} ; Λ ; vect(r∗) )

        vect(a∗) = argmin{ r(vect(a)) : vect(a) ∈ Rh }

        :param Rh: subset of population - array N * 2 - set of 2D-points
        :param weights: array M * 2 - set of 2D-weigthing factor
        :param utopian: 2D-points
        :return:
        """

        if (len(Rh) > 1):
            ra = [self.r2_indicator_optimize(np.delete(Rh, index, axis=0), weights, utopian) for index, a in enumerate(Rh)]

            return Rh[ra.index(min(ra))], ra.index(min(ra))
        else:
            return Rh[0], 0

    def delete_element_in_population(self, element, P):
        """
        Find element in P array and then delete it from P.
        We take the two last components of our Population which is the objective space.
        :param element: element to delete
        :param P: population - numpy array
        :return: new population - numpy array
        """

        return np.delete(P, self.find_index_array(P[:, NORMALIZED_VALUE_LOW_INDEX:], element), axis=0)

    def dominate(self, p1, p2):
        """
        'dominate' means : for each i in p1, f(p1_i) <= f(p2_i) (p1 dominates p2)

        :param p1: array
        :param p2: array
        :return: true if p1 dominates p2, false otherwise
        """

        for i in range(0, p1.shape[0]):
            if (p1[i] > p2[i]):
                return False

        return True

    def find_index_array(self, P, val):
        """
        Find the index of val inside P. We know that val exists in P.
        :param P:
        :param val:
        :return:
        """

        for i, p in enumerate(P):
            if np.array_equal(p, val):
                return i

        raise TypeError("Error: cannot find index of value")

    def fast_nondominated_sort(self, P):
        """
        Fast non dominated sort algorithm
        :param P: population set containing solution - numpy array (initial_population_number,fun dimension)
        :return: pareto front list - list of array list
        """
        Sp = [[] for i in range(0, P.shape[0])]
        Np = np.zeros((P.shape[0],))
        F = []
        F_temp = []

        for cpt, p in enumerate(P):
            for q in P:
                if np.array_equal(p, q):
                    pass
                else:
                    if self.dominate(p, q):
                        Sp[cpt].append(q)
                    elif self.dominate(q, p):
                        Np[cpt] = Np[cpt] + 1
            if Np[cpt] == 0:
                p_rank = 0
                F_temp.append(p)

        F.append(F_temp)

        i = 0
        while len(F[i]) > 0:
            H = []
            for p in F[i]:
                index = self.find_index_array(P, p)
                for k, q in enumerate(Sp[index]):
                    index2 = self.find_index_array(P, q)
                    Np[index2] = Np[index2] - 1
                    if Np[index2] == 0:
                        p_rank = p_rank + 1
                        H.append(q)
            i = i + 1
            if len(H) > 0:
                F.append(H)
            else:
                break

        return F

    def selection_parents(self, population):
        """
        Selection of 2 parents randomly
        :param population: population set containing solution - numpy array (initial_population_number,fun dimension)
        :return: population subset - numpy array (2,2)
        """

        return [population[elmt] for elmt in np.random.choice(len(population), 2, replace=False)]

    def probability_crossover(self, distribution_index):
        """
        Compute beta factor for crossover algorithm
        :param distribution_index: value used to compute beta factor of crossover - positive int
        :return: beta factor
        """

        value = np.random.random_sample()

        return math.pow(2 * value, (1 / (distribution_index + 1))) if value <= 0.5 else math.pow(1 / (2 * (1 - value)), (1 / (distribution_index + 1)))

    def heritage(self, weighting_1, weighting_2, parents, gene):
        """
        Compute children value
        :param weighting_1: weighting parent 1 - 1(+/-)beta
        :param weighting_2: weighting parent 2 - 1(+/-)beta
        :param parents: list of parents
        :param gene: gene to be inherited
        :return: new point - Children
        """

        return 0.5 * (weighting_1 * parents[0][gene] + weighting_2 * parents[1][gene])

    def swap_gene(self, child_1, child_2, gene):
        """
        :param child_1: population member one - numpy array (1,n)
        :param child_2: population member two - numpy array (1,n)
        :param gene: index of the ith-"gene" (ith-value in the list describing the population member)
        :return: children after swapping
        """
        temp = child_1[gene]
        child_1[gene] = child_2[gene]
        child_2[gene] = temp

        return child_1, child_2

    def children(self, population, probability_cross, probability_cross_variation,
                 probability_cross_variation_swap):
        """
        Genetic algorithm - Crossover algorithm (SBX) with random choice of the child (0.5 probability for both)
        :param population: population set containing solution - numpy array (initial_population_number,fun dimension)
        :param probability_cross: value used to compute beta factor of crossover - positive int
        :param probability_cross_variation: probability of crossover event - float in [0,1]
        :param probability_cross_variation_swap: probability of swap gene value between childs - float in [0-1]
        :return: population member
        """
        child_1 = []
        child_2 = []
        parents = self.selection_parents(population)

        for gene in range(len(parents[0])):
            # If variations depending on all parents
            if np.random.random_sample() < probability_cross_variation:
                proba_cross = self.probability_crossover(probability_cross)
                child_1.append(self.heritage(1 + proba_cross, 1 - proba_cross, parents, gene))
                child_2.append(self.heritage(1 - proba_cross, 1 + proba_cross, parents, gene))
            # Else, "simple" copy
            else:
                child_1.append(parents[0][gene])
                child_2.append(parents[1][gene])

            # Swap crossover - Inversion of a gene between children
            if np.random.random_sample() < probability_cross_variation_swap:
                child_1, child_2 = self.swap_gene(child_1, child_2, gene)

        # Random choice of the chosen child
        return random.choice([child_1, child_2])

    def polynomial_proba(self, probability_index):
        """
        Compute Beta factor for mutation algorithm
        :param probability_index: value used to compute beta factor of mutation - positive int
        :return: beta factor
        """
        value = np.random.random_sample()

        return (math.pow(2 * value, 1 / (probability_index + 1)) - 1) if value < 0.5 else (1 - math.pow(2 * (1 - value), 1 / (probability_index + 1)))

    def mutation(self, population, probability_mutation, probability_muta_index, max_fn, min_fn):
        """
        Genetic algorithm - Polynomial mutation algorithm
        :param population: population set containing solution - numpy array (initial_population_number,fun dimension)
        :param probability_mutation: probability of mutation event - float in [0,1]
        :param probability_muta_index: value used to compute beta factor of mutation - positive int
        :param max_fn: highest border of solution - int
        :param min_fn: lowest border of solution - int
        :return: mutated element
        """
        for gene in range(len(population)):
            # If mutation is activated, there is a modification of the current gene else not
            if np.random.random_sample() < probability_mutation:
                population[gene] = population[gene] + (max_fn - min_fn) * self.polynomial_proba(
                    probability_muta_index)

            # Protection against value outside the scope
            if population[gene] > max_fn:
                population[gene] = max_fn
            if population[gene] < min_fn:
                population[gene] = min_fn

        return population

    def crossover_mutation(self, population, probability_cross, probability_cross_variation,
                           probability_cross_variation_swap, probability_mutation,
                           probability_muta_index, max_fn, min_fn):
        """
        Genetic algorithm - SBX crossover and polynoial mutation algorithm
        :param population: population set containing solution - numpy array (initial_population_number,fun dimension)
        :param probability_cross: value used to compute beta factor of crossover - positive int
        :param probability_cross_variation: probability of crossover event - float in [0,1]
        :param probability_cross_variation_swap: probability of swap gene value between childs - float in [0-1]
        :param probability_mutation: probability of mutation event - float in [0,1]
        :param probability_muta_index: value used to compute beta factor of mutation - positive int
        :param max_fn: highest border of solution - int
        :param min_fn: lowest border of solution - int
        :return: new child (SBX crossover and polynomial mutation)
        """
        child = self.children(population, probability_cross, probability_cross_variation,
                              probability_cross_variation_swap)
        child_mutate = self.mutation(child, probability_mutation, probability_muta_index, max_fn, min_fn)

        return child_mutate

    def optimize(self, fun, utopian_point, iteration_number, initial_population_number, probability_cross_index,
                 probability_cross_variation, probability_cross_variation_swap,
                 probability_mutation, probability_muta_index, max_fn, min_fn):
        """
        Main function of R2-EMOA algorithm

        Exemple:
            import cocoex as ex
            suite = ex.Suite("bbob-biobj", "", "")
            for fun in suite:
                R2 = R2_EMOAs(fun)
                R2.optimize(fun, np.array([1e-8, 1e-8]), 10000, 10, 15, 0.9, 0.5, (1/fun.dimension), 20, 100, -100)

        :param fun: coco objectives-functions
        :param utopian_point: Utopian point for reference - numpy array (1,1)
        :param iteration_number: Stopping criterion - positive int
        :param initial_population_number: Initial population size - positive int
        :param probability_cross_index: value used to compute beta factor of crossover - positive int
        :param probability_cross_variation: probability of crossover event - float in [0,1]
        :param probability_cross_variation_swap: probability of swap gene value between childs - float in [0-1]
        :param probability_mutation: probability of mutation event - float in [0,1]
        :param probability_muta_index: value used to compute beta factor of mutation - positive int
        :param max_fn: highest border of solution - int
        :param min_fn: lowest border of solution - int
        :return: population set - numpy array (initial_population_number,fun dimension)
        """

        print("Initializing optimization...")
        begin = time.time()
        weights = self.get_weights_uniform_2m(500)

        print("Initializing population...")
        self.population_list = np.asarray([[random.uniform(fun.lower_bounds[i], fun.upper_bounds[i]) for i in range(len(fun.lower_bounds))] for value in range(initial_population_number)])

        # Adding no normalized and normalized value (objective projection) to the main data structure (self.population_list)
        # Data structure:
        ######################################################################################################################
        ## 0.....................n ## n+1....................................n+2 ## n+3.................................n+4 ##
        ######################################################################################################################
        ## solution (n dimensions) ## No normalized value (objective projection) ## normalized value (objective projection) ##
        ######################################################################################################################
        ## Random initialization   ##  Data for normalization at each iteration  ##    Default value (initialization): 0    ##
        ######################################################################################################################

        self.population_list = np.append(self.population_list, np.asarray([fun(value_population) for value_population in self.population_list]), axis=1)
        self.population_list = np.append(self.population_list, np.asarray([[VALUE_INIT, VALUE_INIT] for value_population in self.population_list]), axis=1)

        # Normalization of fun() results for value between [0,1]
        print("Initial population:")
        print(self.population_list)

        print(" Optimization in progress")

        while True:
            # Creating, normalization and adding offspring in population
            offspring = R2.crossover_mutation(self.population_list[:, :self.population_list.shape[1] - IGNORE_OBJECTIVE_VALUE_INDEX],
                                              probability_cross_index, probability_cross_variation,
                                              probability_cross_variation_swap, probability_mutation,
                                              probability_muta_index, max_fn, min_fn)

            offspring = np.append(offspring, np.asarray(fun(offspring)))
            offspring = np.append(offspring, np.asarray([VALUE_INIT, VALUE_INIT]))
            self.population_list = np.append(self.population_list, [offspring], axis=0)

            # Fast non-dominated sorting algorithm to define pareto front levels
            R = self.fast_nondominated_sort(self.population_list[:, NO_NORMALIZED_VALUE_LOW_INDEX:NO_NORMALIZED_VALUE_HIGH_INDEX])

            # Normalization
            Rh = self.normalize_points(R)

            # R2 comparison and suppression of the worst solution
            a_star, _ = self.argmin_ra(Rh, weights, utopian_point)
            self.population_list = self.delete_element_in_population(a_star, self.population_list)

            # Stopping criterion of the algorithm
            iteration_number = iteration_number - 1
            if iteration_number == 0:
                break

        print("final population")
        print(self.population_list)
        print("time: ", time.time() - begin)
        print("Optimization ending")

        return self.population_list


###################################################################
#                       Test on COCO                              #
###################################################################

if __name__ == '__main__':

    import cocoex as ex

    suite = ex.Suite("bbob-biobj", "", "")
    observer = ex.Observer("bbob-biobj", "result_folder:doctest_OK3")

    for fun in suite:
        print("Number of objectives: ", fun.number_of_objectives)
        if fun.dimension < 40:
            continue
        print(fun.dimension)
        fun.observe_with(observer)
        R2 = R2_EMOAs(fun)
        # Configuration by default from R2 Indicator-Based Multiobjective Search research paper (Dimo Brockhoff)
        # Only the iteration number is changed due to the computation time
        # Tested for 100 - 150 - 500 - 1000 - 5000 - 10000
        # Official results for 150 and 1000 only (tecnology restriction)
        R2.optimize(fun, np.array([-1e-1, -1e-1]),1000 , 10, 15, 0.9, 0.5, (1 / fun.dimension), 20, 100, -100)
