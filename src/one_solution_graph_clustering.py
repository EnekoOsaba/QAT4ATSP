# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:14:08 2020

@author: 108431
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from networkx.algorithms.community import modularity
import networkx as nx
import numpy as np
import pandas as pd
import random


distancesDict = {0:'silhouette', 1:'modularity', 2:'davies_bouldin_score', 3:'calinski_harabasz_score'}
metricsToApply = sorted([0,1,2,3])

generations = 80
populationSize = 20

# swapping operator
def swapping(population, prob = 0.10):
    newPopulation = population.copy()
    for individual in newPopulation:
        swapTimes = 1
        #print('Number of changes in the individual', swapTimes)
        #print('individual', individual)
        for swap in range(swapTimes):
            ## Two exclusive indices, False for replacement so they are exclusive
            x1, x2 = np.random.choice(range(len(individual)), 2, False)
            #print('swapping between', x1, x2, ' changing values', individual[x2], individual[x1])
            ## Swap the genes at the indices
            individual[x1], individual[x2] = individual[x2], individual[x1]
            #print ('new values x1,x2:', individual[x1], individual[x2])
            #print('new individual', individual)
    return newPopulation

# insertion operator
def insertion(population, prob = 0.10):
    newPopulation = population.copy()
    index = 0
    for individual in newPopulation:
        #print('individual', individual)
        newIndividual = individual.copy()
        exist = True
        while exist:
            exist = False
            newIndividual = individual.copy()
            whereToInsert, whatToInsert = sorted(np.random.choice(range(len(individual)), 2, False))
                #print('insertion en position', whereToInsert, 'with value from position', whatToInsert)
            newIndividual = np.insert(newIndividual, whereToInsert, newIndividual[whatToInsert])
            newIndividual = np.delete(newIndividual,whatToInsert+1)
            for i in population:
                if np.array_equal(newIndividual,i):
                    exist = True
                    break
            for i in newPopulation:
                if np.array_equal(newIndividual,i):
                    exist = True
                    break
            exist = False
        newPopulation[index] = newIndividual    
        index = index+1
    
    return newPopulation

#2-OPT operator
def two_opt(population, prob = 0.10):
    newPopulation = population.copy()
    for individual in newPopulation:
        twoOptTimes = 1
        #print('individual', individual)
        for twoOpt in range(twoOptTimes):
            x1, x2 = sorted(np.random.choice(range(len(individual)), 2, False))
            individual[x1:x2] = individual[x1:x2][::-1]
            #print('2-opt - fragment', x1, x2, ' reversing result', individual)
    return newPopulation

# PARAMS: distanceToApply=array of integers (size=population) according to distancesDict
# distancesDict = {0:'silhouette', 1:'modularity', 2:'davies_bouldin_score', 3:'calinski_harabasz_score'}
def _fitness(distances_df, points, population, distanceToApply, populationSize = populationSize, best=True):       
    scores = []       
    for individualIdx in range(len(population)):
        individual = population[individualIdx]
        if distancesDict[distanceToApply]=='silhouette':
            #The best value is 1 and the worst value is -1
            scores.append(silhouette_score(distances_df.to_numpy(), individual, metric = 'precomputed'))   
        elif distancesDict[distanceToApply]=='modularity':
            # range: [−1/2,1) Best value is 1
            scores.append(modularity(nx.from_numpy_matrix(distances_df.to_numpy()), [np.where(individual==cluster)[0] for cluster in set(individual)], weight='weight'))
        # lower values indicating better clustering
        elif distancesDict[distanceToApply]=='davies_bouldin_score':
            scores.append(-davies_bouldin_score(points, individual))
        # biggest values indicating better clustering
        elif distancesDict[distanceToApply]=='calinski_harabasz_score':
            scores.append(calinski_harabasz_score(points, individual))
    if best:
        selectedScoresIdx = np.argsort(scores)[::-1][:populationSize]      
    else:
        selectedScoresIdx = np.argsort(scores)[:populationSize]      
    return selectedScoresIdx, np.max(scores) if best else np.min(scores)
    
def reduction(distances_df, points, wholePopulation, metricsToApply, populationSize = populationSize, end = False): 
    newPopulation = []    

    fittest, bestScore = _fitness(distances_df, points, wholePopulation, metricsToApply, populationSize)  
    for item in fittest: 
        newPopulation.append(wholePopulation[item])          
    return np.array(newPopulation), bestScore 

def solvePartitionProblem(cluster_size,tsp_matrix,node_array):

    cluster_size_threshold = cluster_size
    
    # Problem statement
    
    numItems = len(tsp_matrix)
    points = node_array
    #G = nx.from_numpy_matrix(tsp_matrix)
    distances_df = pd.DataFrame(tsp_matrix)
     
       
    # PROBLEM CONFIG
    population = []  
    numClusters = int(numItems / cluster_size_threshold) if (numItems % cluster_size_threshold)==0 else int(numItems / cluster_size_threshold) + 1
    numItemsPerCluster_dict = {cluster: [cluster]*(int(numItems / numClusters) + 1) if cluster<= (numItems % numClusters) else [cluster]*(int(numItems / numClusters)) for cluster in range(1, numClusters + 1)}
    vocabulary = np.array([value for valueList in list(numItemsPerCluster_dict.values()) for value in valueList]) # possible values for the individuals
    
    for _ in range(populationSize): 
        random.shuffle(vocabulary) 
        population.append(vocabulary.copy())
    population = np.array(population)
    
    # generations
    scoreEvolution = {metric:[] for metric in metricsToApply}
    for _ in range(generations):
        
        #print(_)
        
        newPopulation = []
        
#        newSubpopulation_twoopt = two_opt(population, prob=1/numItems) 
        newSubpopulation_insertion = insertion(population, prob=1/numItems)
#        newSubpopulation_swap = swapping(population, prob=1/numItems)
        #newSubPopulation, bestScore = reduction(distances_df, points, np.concatenate((population, newSubpopulation_twoopt, newSubpopulation_insertion, newSubpopulation_swap)), metricsToApply[1], populationSize) 
        newSubPopulation, bestScore = reduction(distances_df, points, np.concatenate((population, newSubpopulation_insertion)), metricsToApply[3], populationSize) 
        
        scoreEvolution[metricsToApply[3]].append(bestScore)
        newPopulation.append(newSubPopulation)
            
        population = np.concatenate(newPopulation)
    
    winner, bestScore = reduction(distances_df, points, population, metricsToApply[3], populationSize) 
    print('Winners {0} with scores {1}'.format(winner[1],bestScore))

    return winner[0]