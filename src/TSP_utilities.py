import numpy as np
import csv
import sys

def readInstance(instance):
    """
    Creates array from a TSP instance.
    """
    if instance=='data/ftv35.atsp':
        try:
            reader = csv.reader(open('data/ftv38.atsp'), delimiter=' ')
        except FileNotFoundError:
            sys.exit("Fichero param.config no encontrado")
    elif instance=='data/ftv33.atsp':
        try:
            reader = csv.reader(open('data/ftv38.atsp'), delimiter=' ')
        except FileNotFoundError:
            sys.exit("Fichero param.config no encontrado")
    else:
        try:
            reader = csv.reader(open(instance), delimiter=' ')
        except FileNotFoundError:
            sys.exit("Fichero param.config no encontrado")

    weight_list = []
    
    weight=False   
    
    for row in reader:
        
        if row[0]=="EOF":
            break
        if weight==True:
            weights = row[0].split(",")
            weight_list.append([float(i) for i in weights])
            
        if row[0]=="EDGE_WEIGHT_SECTION":
           weight=True
               
    if instance=='data/ftv35.atsp':
        #print('CUT THE INSTANCE TO ftv35')
        del weight_list[12]
        del weight_list[11]
        del weight_list[8]
        for item in weight_list:
            del item[12]
            del item[11]
            del item[8]
        
    if instance=='data/ftv33.atsp':
        #print('CUT THE INSTANCE TO ftv33')
        del weight_list[34]
        del weight_list[25]
        del weight_list[15]
        del weight_list[8]
        del weight_list[4]
        for item in weight_list:
            del item[34]
            del item[25]
            del item[15]
            del item[8]
            del item[4]
        
            
    return weight_list


def get_tsp_matrix(weight_array):
    """
    Creates distance matrix out of given coordinates.
    
    """
    
    number_of_nodes = len(weight_array)
    matrix = np.zeros((number_of_nodes, number_of_nodes))
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            matrix[i][j] = float(weight_array[i][j]) * 1.0
    return matrix

def get_tsp_reduced_matrix(weight_array,index):
    """
    Creates distance matrix out of given coordinates.
    
    """    
    number_of_nodes = len(weight_array)
    matrix = np.zeros((len(index), len(index)))
    aux_i = -1
    aux_j = -1
    for i in range(number_of_nodes):
        if i in index:
            aux_i = aux_i+1
            aux_j = -1
            for j in range(number_of_nodes):
                if j in index:
                    aux_j = aux_j+1
                    matrix[aux_i][aux_j] = float(weight_array[i][j]) * 1.0
                    
    return matrix


def calculate_cost(cost_matrix, solution):
    cost = 0
    for i in range(len(solution)):
        a = i%len(solution)
        b = (i+1)%len(solution)
        cost += cost_matrix[solution[a]][solution[b]]

    return cost

def points_order_to_binary_state(points_order):
    """
    Transforms the order of points from the standard representation: [0, 1, 2],
    to the binary one: [1,0,0,0,1,0,0,0,1]
    """
    number_of_points = len(points_order)
    binary_state = np.zeros((len(points_order) - 1)**2)
    for j in range(1, len(points_order)):
        p = points_order[j]
        binary_state[(number_of_points - 1) * (j - 1) + (p - 1)] = 1
    return binary_state

def binary_state_to_points_order(binary_state):
    """
    Transforms the the order of points from the binary representation: [1,0,0,0,1,0,0,0,1],
    to the binary one: [0, 1, 2]
    """
    points_order = []
    number_of_points = int(np.sqrt(len(binary_state)))
    for p in range(number_of_points):
        for j in range(number_of_points):
            if binary_state[(number_of_points) * p + j] == 1:
                points_order.append(j)
    return points_order
