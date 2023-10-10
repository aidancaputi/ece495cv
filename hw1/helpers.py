import math, random
#import numpy as np

def dot(matrix1, matrix2):

    #print("dotting")
    #print_size(matrix1)
    #print_size(matrix2)
    if(len(matrix1) == len(matrix2)):
        '''print_size(matrix1)
        print_size(matrix2)'''
        result = elementwise_multiply(matrix1, matrix2)
        #num = np.multiply(matrix1, matrix2)

    else:
        result = [[0.0]*len(matrix2[0]) for i in range(len(matrix1))]
        #num = np.dot(matrix1, matrix2)

        #print_size(result)

        for i in range(len(result)):
            for j in range(len(result[0])):
                for k in range(len(matrix1[0])):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    '''print("my dot:", result)
    print("numpy dot:", num)
    np.testing.assert_allclose(result, num, rtol=1e-5, atol=0)'''

    return result

def transpose(matrix):
    #print("original matrix", matrix)
    orig_rows = len(matrix)
    orig_cols = len(matrix[0])
    
    transposed = [[0]*orig_rows for i in range(orig_cols)]

    for i in range(orig_rows):
        for j in range(orig_cols):
            transposed[j][i] = matrix[i][j]
    #print("transposed", transposed)
    return transposed

#scale a vector element-wise by a constant
def ktimesv(k, u): 
    #print("scaling k=", k, "to matrix:", u)

    scaled = []

    for i in range(len(u)):
        new_row = []
        for j in range(len(u[0])):
            new_row.append(u[i][j] * k)
        scaled.append(new_row)

    return scaled

def elementwise_multiply(u, v):

    if((len(u) != len(v)) or (len(u[0]) != len(v[0]))):
        print("matrices must be same size to multiply elementwise")
        print_size(u)
        print_size(v)
        print("are not the same size")
        exit(0)

    output = []
    for i in range(len(u)):
        new_row = []
        for j in range(len(u[i])):
            new_row.append(u[i][j] * v[i][j])
        output.append(new_row)
    
    return output

def elementwise_add(u, v):

    if((len(u) != len(v)) or (len(u[0]) != len(v[0]))):
        print("matrices must be same size to add elementwise")
        print_size(u)
        print_size(v)
        print("are not the same size")
        exit(0)

    output = []
    for i in range(len(u)):
        new_row = []
        for j in range(len(u[i])):
            new_row.append(u[i][j] + v[i][j])
        output.append(new_row)
    
    return output

def elementwise_sub(u, v):

    if((len(u) != len(v)) or (len(u[0]) != len(v[0]))):
        print("matrices must be same size to subtract elementwise")
        print_size(u)
        print_size(v)
        print("are not the same size")
        exit(0)
    
    output = []
    for i in range(len(u)):
        new_row = []
        for j in range(len(u[i])):
            new_row.append(u[i][j] - v[i][j])
        output.append(new_row)
    
    return output

#generates a matrix with random numbers between 0 and 1
def generate_random_matrix(n_rows, n_cols):
    matrix = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            row.append(random.uniform(-1, 1))
        matrix.append(row)
    return matrix

def print_size(matrix):
    if(isinstance(matrix[0], float)):
        print("(", len(matrix), ",", 1, ")")
    else: 
        print("(", len(matrix), ",", len(matrix[0]), ")")

#simply prints all the info about the network being trained
def print_train_status(dataset, n_layers, input_dim, output_dim, hidden_units, learning_rate, epochs, split):

    print("\n**************** Training a MLP with the following details ****************")
    #print(" | dataset | number of layers | hidden units per layer | learning rate | epochs | train/test split |")
    print(f" | {dataset} dataset | {n_layers} hidden layers | {hidden_units} units per hidden layer | learning rate:{learning_rate} | {epochs} epochs | train/test split:{split} |")

    return