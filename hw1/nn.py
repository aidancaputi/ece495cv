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

#multiple input and weights, then add bias
def fc_linear_layer(x, w, b):
    return vector_add(dot(x, w), b)

#sigmoid function element wise to a vector
def sigmoid(vec):
    return [[(1/(1 + (math.exp(-1 * x)))) for x in tt] for tt in vec]

#single layer perceptron
def slp(x, w, b):
    return sigmoid(fc_linear_layer(x, w, b))

#multi-layer perceptron
def mlp(x, w_vec, b_vec):

    cache = []

    #make sure weights and biases vectors are same length
    if(len(w_vec) != len(b_vec)):
        print("Error in mlp() -> not the same amount of weights and biases!")
        exit(1)
    
    #start with input x value
    out = x

    #loop through weights and bias and apply nested single layer perceptron starting from the first
    for w, b in zip(w_vec, b_vec):
        out = slp(out, w, b)
        cache.append(out)

    return out, cache

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

#take the hyperparameters and initialize the weights
def initialize_weights(n_layers, hidden_units, input_dim, output_dim):

    weights = []

    for i in range(n_layers):

        #this is the input layer weights
        if(i == 0):
            weights.append(generate_random_matrix(hidden_units, input_dim))
        
        #this is a middle hidden layer (i.e., not the output layer)
        elif((i > 0) and (i < (n_layers - 1))):
            weights.append(generate_random_matrix(hidden_units, hidden_units))

        #this is the output layer
        else:
            weights.append(generate_random_matrix(hidden_units, output_dim))
    
    return weights

#generates a matrix with random numbers between 0 and 1
def generate_random_matrix(n_rows, n_cols):
    matrix = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            row.append(random.uniform(-1, 1))
        matrix.append(row)
    return matrix

def initialize_biases(n_layers, hidden_units, output_dim):
    biases = []

    #for each layer
    for i in range(n_layers):
        
        #create bias vector with same size as output of that layer
        #this is the input layer weights
        if(i < (n_layers - 1)):
            biases.append([[random.uniform(-1, 1) for x in range(hidden_units)]])

        #this is the output layer
        else:
            biases.append([[random.uniform(-1, 1) for x in range(output_dim)]])
    #print("bioas:", biases)
    return biases

#simply prints all the info about the network being trained
def print_train_status(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split):

    print("\nTraining MLP network using the following data:")
    print("X data: ", X)
    print("y data: ", y)

    print("\nAnd the following parameters:")
    print("Number of layers: ", n_layers)
    print("Input dimensions: ", input_dim)
    print("Output dimensions: ", output_dim)
    print("Number of hidden units in each layer: ", hidden_units)
    print("Learning rate: ", learning_rate)
    print("Train/test split: ", train_test_split)

    return

def split(X, y, train_test_split):

    #make sure X and y data have matching sizes
    if(len(X) != len(y)):
        print("ERROR: X and y are not the same size. Check your training data to make sure every X has a y.")
        exit(1)

    X_train = X
    y_train = y
    X_test = []
    y_test = []

    num_samples = len(X)

    train_size = int(train_test_split * num_samples)
    test_size = num_samples - train_size

    for i in range(test_size):

        #randomly choose an index in X_train
        chosen_idx = random.randint(0, len(X_train) - 1)

        #pop the X_train and y_train values at that index 
        popped_X = X_train.pop(chosen_idx)
        popped_y = y_train.pop(chosen_idx)

        #add the popped values to the test data
        X_test.append(popped_X)
        y_test.append(popped_y)

    return X_train, y_train, X_test, y_test

#loss function
def squared_loss(y, pred):
    
    # (u - v) * (u - v)
    u_minus_v = vector_subtraction(y, pred)
    return dot(u_minus_v, u_minus_v)

#this must return something that is the same size as weights
def squared_loss_gradient(weights, bias, current_layer_input, wildcard, mode):

    '''print("\nEnter gradient with mode", mode)
    print("weights:", weights)
    print("bias:", bias)
    print("layer input", current_layer_input)
    print("wildcard", wildcard)'''

    sig = slp(current_layer_input, weights, bias)

    one_minus_sig = [[(1-element) for element in row] for row in sig]

    sig_dot_sig = dot(sig, one_minus_sig)

    b = dot(sig_dot_sig, wildcard)

    #if we are computing weight gradients, we have to multiply by the output of the previous layer
    if(mode == "weight"):
        #print("resulting gradient: ", dot(b, current_layer_input))
        return dot(b, current_layer_input)
    
    #if we are computing bias gradient, we dont need to multiply by the output of the previous layer
    #print("resulting gradient: ", b)
    return b

#train the network
def train(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split, epochs):

    #log the details of this training
    print_train_status(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split)

    #use hyperparameters to initialize weights for the network
    weights = initialize_weights(n_layers, hidden_units, input_dim, output_dim)
    
    #use hyperparameters to initialize biases to 0
    biases = initialize_biases(n_layers, hidden_units, output_dim)
    #print(*biases, sep='\n')

    X_train, y_train, X_test, y_test = split(X, y, train_test_split)

    print("training X: ", X_train)
    print("training y: ", y_train)
    print("test X: ", X_test)
    print("test y: ", y_test)

    print("\ninital weights:")
    print(*weights, sep='\n')
    print("\ninital biases:")
    print(*biases, sep='\n')

    best_loss = 1.0
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1, 1):

        #losses and num correct predictions for this epoch
        loss_total = 0.0

        #print(weights)

        #for each training X (i.e, an epoch)
        for cur_X, cur_y in zip(X_train, y_train):

            #perform forward pass and add prediction to predictions vector
            #print("forward passing")
            pred, cache = mlp(cur_X, weights, biases)

            #print("prediction:", pred, "expected:", cur_y)
            #print("cached: ", cache)

            #compute loss and append it to the losses array
            loss_total += squared_loss(cur_y, pred)[0][0]

            #print("performing back pass")

            #BACKPROPAGATION
            prev_grads = tuple()

            #for every layer in the network
            for layer in range(n_layers - 1, -1, -1):

                #hidden to output layer
                if(layer == (n_layers - 1)):

                    #print("\nold values for weights and biases in output layer:", weights[layer], biases[layer])
                    weight_grad = transpose(squared_loss_gradient(weights[layer], biases[layer], cache[layer - 1], ktimesv(2, vector_subtraction(cur_y, pred)), mode="weight"))
                    weights[layer] = vector_subtraction(weights[layer], ktimesv(learning_rate, weight_grad))
                    
                    bias_grad = transpose(squared_loss_gradient(weights[layer], biases[layer], cache[layer - 1], ktimesv(2, vector_subtraction(cur_y, pred)), mode="bias"))
                    biases[layer] = vector_subtraction(biases[layer], ktimesv(learning_rate, bias_grad))

                    prev_grads = (weight_grad, bias_grad)
                    
                    #print("gradients computed for output layer              : ", prev_grads[0], prev_grads[1])
                    print("new values for weights and biases in output layer:", weights[layer], biases[layer])
                
                #hidden
                else:

                    #print("\nold values for weights and biases in hidden layer:", weights[layer], biases[layer])
                    weight_grad = transpose(squared_loss_gradient(weights[layer], biases[layer], cur_X, dot(weights[layer + 1], prev_grads[1]), mode="weight"))
                    weights[layer] = vector_subtraction(weights[layer], ktimesv(learning_rate, weight_grad))

                    bias_grad = transpose(squared_loss_gradient(weights[layer], biases[layer], cur_X, dot(weights[layer + 1], prev_grads[1]), mode="bias"))
                    biases[layer] = vector_subtraction(biases[layer], ktimesv(learning_rate, bias_grad))

                    prev_grads = (weight_grad, bias_grad)
                    
                    #print("gradients computed for hidden layer              :", prev_grads[0], prev_grads[1])
                    print("new values for weights and biases in hidden layer:", weights[layer], biases[layer])



                
        print("epoch", epoch, "loss:", (loss_total / len(X_train)))
        '''print("weights:")
        print(*weights, sep='\n')
        print("biases:")
        print(*biases, sep='\n')'''

        #update best loss and accuracy for model if this epoch bested them
        if((loss_total / len(X_train)) < best_loss):
            best_loss = (loss_total / len(X_train))

    print('\nAfter training:')
    print("Best training accuracy:", best_accuracy)
    print("Best loss:", best_loss)
    print("Final weights:")
    print(*weights, sep='\n')
    print("Final biases:")
    print(*biases, sep='\n')

    return