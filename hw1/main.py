from nn import *
from datasets import *

def main():

    #should be able to try changing combinations of the following:
        #number of layers
        #number of units in each hidden layer
        #learning rate
        #train test split
        #epochs
    datasets_to_try = ['xor', 'adder']
    splits_to_try = [0.5, 0.75, 0.9]
    hyperparameters_to_try = [(1, 5), (2, 10), (3, 20)] #these are in the form (number of hidden layers, number of units per layer)
    learning_rates_to_try = [0.001, 0.01, 0.1]
    epochs_to_try = [10, 100, 1000]

    adder_test_losses = {}

    #train(X_train, y_train, X_test, y_test, n_layers, input_dim, output_dim, hidden_units, learning_rate, num_epochs, train_test_split)

    best_loss = 1.0
    best_hyperparameters = (0,0)
    for hyperparameters in hyperparameters_to_try:
        loss = train2('adder', hyperparameters, 0.01, 100, 0.8)
        if(loss < best_loss):
            best_loss = loss
            best_hyperparameters = hyperparameters

    print("best loss for hyperparameters test was:", best_loss, "with the parameters", best_hyperparameters)
    
    best_loss = 1.0
    best_lr = 0.0
    for lr in learning_rates_to_try:
        loss = train2('adder', best_hyperparameters, lr, 100, 0.8)
        if(loss < best_loss):
            best_loss = loss
            best_lr = lr

    print("best loss for learning rate test was:", best_loss, "with a learning rate of", best_lr)

    best_loss = 1.0
    best_epochs = 0.0
    for epochs in epochs_to_try:
        loss = train2('adder', best_hyperparameters, best_lr, epochs, 0.8)
        if(loss < best_loss):
            best_loss = loss
            best_epochs = epochs

    print("best loss for epochs test was:", best_loss, "with", best_epochs, "epochs")

    best_loss = 1.0
    best_split = 0.0
    for split in splits_to_try:
        loss = train2('adder', best_hyperparameters, best_lr, best_epochs, split)
        if(loss < best_loss):
            best_loss = loss
            best_split = split

    print("best loss for training split test was:", best_loss, "with training split", best_split)
    
    print("\nso, the best network for the adder dataset has the following details:")
    print("hidden layers:", best_hyperparameters[0])
    print("units per hidden layer:", best_hyperparameters[1])
    print("learning rate:", best_lr)
    print("epochs:", best_epochs)
    print("train/test split:", best_split)
    train2('adder', best_hyperparameters, best_lr, best_epochs, best_split)


if __name__ == "__main__":
    main()