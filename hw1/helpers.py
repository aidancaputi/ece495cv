import math, random

def dot(matrix1, matrix2):

    #print("doing dot prod with ", matrix1, matrix2)

    if((len(matrix1[0]) == len(matrix2[0])) and (len(matrix1) == len(matrix2))):
        matrix2 = transpose(matrix2)
    
    if((len(matrix1[0]) == 1) and (len(matrix1) == 1)):
        return ktimesv(matrix1[0][0], matrix2)
    
    if((len(matrix2[0]) == 1) and (len(matrix2) == 1)):
        return ktimesv(matrix2[0][0], matrix1)
    
    if len(matrix1[0]) != len(matrix2):
        print("wrong size matrices, can't dot product")
        exit(1)

    # Get the dimensions of the matrices
    m1 = len(matrix1)
    n1 = len(matrix1[0])
    m2 = len(matrix2)
    n2 = len(matrix2[0])

    #resulting matrix is 
    out = [[0]*n2 for i in range(m1)]

    for i in range(m1):
        for j in range(n2):
            for k in range(n1):
                out[i][j] += matrix1[i][k] * matrix2[k][j]

    #print("result was: ", out)

    return out

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

def vector_add(u, v):

    #print("adding ", u, v)
    if(len(u) != len(v)):
        print()
    output = []
    for i in range(len(u)):
        new_row = []
        for j in range(len(u[i])):
            new_row.append(u[i][j] + v[i][j])
        output.append(new_row)
    #print("subtraction was ", output)
    return output

def vector_subtraction(u, v):

    #print("subtracting ", u, v)
    if(len(u) != len(v)):
        print()
    output = []
    for i in range(len(u)):
        new_row = []
        for j in range(len(u[i])):
            new_row.append(u[i][j] - v[i][j])
        output.append(new_row)
    #print("subtraction was ", output)
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