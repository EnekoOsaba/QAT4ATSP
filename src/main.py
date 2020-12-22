import time
import TSP_utilities
from dwave_tsp_solver import DWaveTSPSolver
import graph_clustering,merge_partialTSPs
from datetime import datetime
import random
import numpy as np
import sys

def solveTSPinstanceQTA(instance,name):
    cluster_size_threshold = 20
    weight_array = TSP_utilities.readInstance(instance)
    tsp_matrix = TSP_utilities.get_tsp_matrix(weight_array)
    max_evaluations = 120
    routes_dict = {}
    partitions_dict = {}

    #print("We start here with our local search method. So first, we calculate the initialization cluster")
        
    preliminary_partitions = graph_clustering.solvePartitionProblem(cluster_size_threshold,tsp_matrix,weight_array)
    
    #print(preliminary_partitions)
    
    unique_partitions_id = []
    unique_partitions = []

    for key, value in preliminary_partitions.items():
        if str(value) not in unique_partitions_id:
            unique_partitions_id.append(str(value))
            unique_partitions.append(value)
        
    
    bestPermutation = []
    bestPartition = []
    bestCost = 1000000000
    
    for partitions in unique_partitions:
        bestCost_aux = 1000000000
        currentCost = 100000000
        bestPermutation_aux = []
        evaluations = 0
        while evaluations<max_evaluations/len(unique_partitions):                  
            if evaluations>0:
                newSolution = False
                while newSolution == False:
                    currentPartition = partitions.copy()
                    index1 = random.randint(0, len(partitions)-1)
                    firstElement = currentPartition[index1]
                    index2 = random.randint(0, len(partitions)-1)
                    secondElement = currentPartition[index2]
                
                    while firstElement == secondElement:
                        index2 = random.randint(0, len(partitions)-1)
                        secondElement = currentPartition[index2]
                
                    currentPartition[index1] = secondElement
                    currentPartition[index2] = firstElement
                                        
                    if str(currentPartition) in partitions_dict:
                        newSolution = False
                    else:
                        newSolution = True
                        partitions_dict[str(currentPartition)] = True   
            else:
                currentPartition = partitions.copy()
         
            permutation_route, cost, evaluations = calculatePartitionCost(currentPartition,weight_array,tsp_matrix,evaluations,routes_dict)
            if evaluations==0:
                    evaluations = evaluations+1             
            
            if cost<currentCost:
                currentCost = cost
                partitions = currentPartition.copy()
                bestPartition = currentPartition.copy()
                
            if cost<bestCost_aux:
                bestCost_aux = cost
                bestPermutation_aux = permutation_route
            
            #print('BEST PARTITION', partitions, bestPermutation_aux, bestCost_aux)
                
                
        
        
        if bestCost_aux<bestCost:
            bestCost = bestCost_aux
            bestPermutation = bestPermutation_aux      
        
    #print('BEST PARTITION', partitions, bestPermutation, bestCost)
    print(bestCost)
    #print()
    
    #TSP_utilities.plot_solution(name + str(datetime.now().timestamp()), nodes_array, bestPermutation)
    
    #paintPartition(bestPartition,nodes_array,tsp_matrix,routes_dict)
        
 
def calculatePartitionCost(partitions,weight_array,tsp_matrix, evaluations, routes_dict):
    
    sapi_token = 'DEV-9f670414561136c93e3c450825322d72bd0eea42'
    #sapi_token = 'DEV-2a8221a9290a3004418159b2cc3e39c4f4ac5a58'
    dwave_url = 'https://cloud.dwavesys.com/sapi'
    
    index = []
    route_aux = [] 
    list_index = []
    list_of_solutions = []

    clusters = max(partitions)
        
    #print("We calculate now the optimal tour of each cluster")
    #print(partitions)
    
    for i in range(clusters):
        cluster = i+1
        index = [i for i, e in enumerate(partitions) if e == cluster]
        list_index.append(index)
        if str(index) in routes_dict:
            dwave_solution  = routes_dict[str(index)]
        else:
            reduced_tsp_matrix = TSP_utilities.get_tsp_reduced_matrix(weight_array,index)
            dwave_solver = DWaveTSPSolver(reduced_tsp_matrix, sapi_token=sapi_token, url=dwave_url)
            #dwave_solution = dwave_solver.solve_tsp_DWAVE()
            dwave_solution = dwave_solver.solve_tsp_QBSolv_Tabu()
            #print(TSP_utilities.calculate_cost(reduced_tsp_matrix, dwave_solution))
            evaluations = evaluations + 1
            routes_dict[str(index)] = dwave_solution
            
        route_aux = [index[i] for i in dwave_solution]
        list_of_solutions.append(route_aux)
    
    #print('Time to merge')
    solution = merge_partialTSPs.recomposeTSPsubcycles(weight_array,tsp_matrix,list_index,list_of_solutions)
    route = list(solution[0])
    cost = solution[1]
    permutation_route = []
    permutation_route.append(route[0][0])
    permutation_route.append(route[0][1])
    del route[0]
    while len(route)>1:
        for sublist in route:
            if permutation_route[len(permutation_route)-1] in sublist:
                sublist_aux = list(sublist)
                sublist_aux.remove(permutation_route[len(permutation_route)-1])
                permutation_route.append(sublist_aux[0])
                route.remove(sublist)
                break
        
    #print(permutation_route, cost)     
    
    return permutation_route, cost, evaluations
        
def solveTSPinstanceNOClustering(instance,name):
    weight_array = TSP_utilities.readInstance(instance)
    tsp_matrix = TSP_utilities.get_tsp_matrix(weight_array)
    sapi_token = 'DEV-9f670414561136c93e3c450825322d72bd0eea42'
    #sapi_token = 'DEV-2a8221a9290a3004418159b2cc3e39c4f4ac5a58'
    dwave_url = 'https://cloud.dwavesys.com/sapi'

    if sapi_token is None or dwave_url is None:
        print("You cannot run code on DWave without specifying your sapi_token and url")
    elif len(weight_array) >= 100:
        print("This problem size is to big to run on D-Wave.")
    else:
        #print("DWave solution")      
        dwave_solver = DWaveTSPSolver(tsp_matrix, sapi_token=sapi_token, url=dwave_url)
        #dwave_solution = dwave_solver.solve_tsp_QBSolv_DWAVE()
        dwave_solution = dwave_solver.solve_tsp_QBSolv_Tabu()
        #dwave_solution = dwave_solver.solve_tsp_DWAVE()
        solution_cost = TSP_utilities.calculate_cost(tsp_matrix, dwave_solution)
        #print("DWave:", dwave_solution, solution_cost)
        print(solution_cost)

if __name__ == '__main__':
          
    # solveTSPinstanceNOClustering("data/br17.atsp","QBSolv_br17_")
    # solveTSPinstanceQTA("data/br17.atsp","QTA_br17_")
    
    # for i in range(20):  
    #     solveTSPinstanceNOClustering("data/ftv33.atsp","QBSolv_br17_")
        
    #for i in range(20):  
    #    solveTSPinstanceQTA("data/ftv33.atsp","QTA_br17_")
        
    # for i in range(20):  
    #     solveTSPinstanceNOClustering("data/ftv35.atsp","QBSolv_br17_")
        
    # for i in range(20):  
    #     solveTSPinstanceQTA("data/ftv35.atsp","QTA_br17_")
        
    # for i in range(20):  
    #     solveTSPinstanceNOClustering("data/ftv38.atsp","QBSolv_br17_")
    
    # for i in range(20):  
    #     solveTSPinstanceQTA("data/ftv38.atsp","QTA_br17_")
    
    for i in range(20):  
        solveTSPinstanceNOClustering("data/p43.atsp","QBSolv_br17_")
    
    # for i in range(20):  
    #     solveTSPinstanceQTA("data/p43.atsp","QTA_br17_")
    
    # for i in range(20):  
    #     solveTSPinstanceNOClustering("data/ry48p.atsp","QBSolv_br17_")
    
    # for i in range(20):  
    #     solveTSPinstanceQTA("data/ry48p.atsp","QTA_br17_")

    