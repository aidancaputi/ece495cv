import math, random
from helpers import *
from datasets import *

#multiple input and weights, then add bias
def fc_linear_layer(x, w, b):

    output = dot(x, w)
    
    for i in range(len(dot(x, w))):

        output[i] = elementwise_add([output[i]], [b])[0]

    return output

#sigmoid function element wise to a vector
def sigmoid(vec):
    #return 1 / (1 + np.exp(-vec))
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

#take the hyperparameters and initialize the weights
def initialize_weights(n_layers, hidden_units, input_dim, output_dim):

    weights = []

    for i in range(n_layers):

        #this is the input layer weights
        if(i == 0):
            weights.append(generate_random_matrix(input_dim, hidden_units))
        
        #this is a middle hidden layer (i.e., not the output layer)
        elif((i > 0) and (i < (n_layers - 1))):
            weights.append(generate_random_matrix(hidden_units, hidden_units))

        #this is the output layer
        else:
            weights.append(generate_random_matrix(hidden_units, output_dim))
    
    return weights

def initialize_biases(n_layers, hidden_units, output_dim):
    biases = []

    #print(hidden_units, n_layers)

    #for each layer
    for i in range(n_layers):
        
        #create bias vector with same size as output of that layer
        #this is the input layer weights
        if(i < (n_layers - 1)):
            biases.append([random.uniform(-1, 1) for x in range(hidden_units)])

        #this is the output layer
        else:
            biases.append([random.uniform(-1, 1) for x in range(output_dim)])
    #print("bioas:", biases)
    return biases

#loss function
def loss_fn(y, pred):
    
    # (u - v) * (u - v)
    u_minus_v = elementwise_sub(y, pred)
    return dot(u_minus_v, u_minus_v)

def cost(y, pred):
    
    return elementwise_sub(y, pred)

def sigmoid_derivative(sig):
    one_minus_sig = [[(1-element) for element in row] for row in sig]
    return dot(sig, one_minus_sig)

#this must return something that is the same size as weights
def gradient(mode, a, b):
    


    return

def step(val):
    if(val >= 1.0):
        return 1
    return 0

def train2(dataset, hyperparameters, learning_rate, num_epochs, train_test_split):

    X, y = generate_dataset2(dataset)
    X_train, y_train, X_test, y_test = train_test_slit(X, y, train_test_split)

    n_hid_layers = hyperparameters[0] #number of hidden layers
    n_layers = n_hid_layers + 1
    input_dim = len(X[0]) #input dimensions
    output_dim = len(y[0]) #output dimensions
    hidden_units = hyperparameters[1] #hidden units in each layer
    
    print_train_status(dataset, n_hid_layers, input_dim, output_dim, hidden_units, learning_rate, num_epochs, train_test_split)

    weights = initialize_weights(n_layers, hidden_units, input_dim, output_dim)
    
    biases = initialize_biases(n_layers, hidden_units, output_dim)
    #print(biases)

    for epoch in range(1, num_epochs + 1):

        #print("training... epoch", epoch)
        
        #front pass
        out, cache = mlp(X_train, weights, biases)

        #print("epoch:", epoch, " -> loss:", loss_fn(y_train, out)[0])
        
        prev_grads = tuple()

        for layer in range(n_layers - 1, -1, -1):

            #hidden to output layer
            if(layer == (n_layers - 1)):
                output_grad = dot(cost(out, y_train), sigmoid_derivative(cache[layer]))
                weights[layer] = elementwise_sub(weights[layer], ktimesv(learning_rate, dot(transpose(cache[layer - 1]), output_grad)))
                for row in ktimesv(learning_rate, output_grad):
                    biases[layer] = elementwise_sub([biases[layer]], [row])[0]
                prev_grads = (output_grad, 0)

            #hidden to hidden layer
            elif((layer != (n_layers - 1)) and (layer != 0)):
                hid_grad = dot(dot(prev_grads[0], transpose(weights[layer + 1])), sigmoid_derivative(cache[layer]))
                weights[layer] = elementwise_sub(weights[layer], ktimesv(learning_rate, dot(transpose(cache[layer - 1]), hid_grad))) 
                for row in ktimesv(learning_rate, hid_grad):
                    biases[layer] = elementwise_sub([biases[layer]], [row])[0]
                #biases[layer] = elementwise_sub(biases[layer], ktimesv(learning_rate, hid_grad))
                prev_grads = (hid_grad, 0)

            #input to hidden layer
            else:
                input_grad = dot(dot(prev_grads[0], transpose(weights[layer + 1])), sigmoid_derivative(cache[layer]))
                weights[layer] = elementwise_sub(weights[layer], ktimesv(learning_rate, dot(transpose(X_train), input_grad))) 
                for row in ktimesv(learning_rate, input_grad):
                    biases[layer] = elementwise_sub([biases[layer]], [row])[0]
                #biases[layer] = elementwise_sub(biases[layer], ktimesv(learning_rate, input_grad)) 
        
    #print(biases)

    accurate = 0
    test_losses = []
    predictions = []

    for x, y in zip(X_test, y_test):
        in_x = [x]
        pred, cache = mlp(in_x, weights, biases)
        test_losses.append(loss_fn([y], pred)[0][0])
        predictions.append(pred[0][0])
        #print(pred[0][0], y[0])
        if(int(pred[0][0] > 0.5) == y[0]):
            accurate += 1

    #print("final accuracy: ", float(accurate) / float(len(X_test)))
    print("final test loss:", sum(test_losses) / len(predictions))

    return sum(test_losses) / len(predictions)