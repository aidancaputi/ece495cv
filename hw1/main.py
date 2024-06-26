from nn import *
from datasets import *
import time


def main():

    #start time
    t0 = time.time()

    #establish the different parameters we want to test
    splits_to_try = [0.5, 0.6, 0.75]
    hyperparameters_to_try = [(1, 5), (1, 10), (2, 5), (2, 10), (3, 5)] #these are in the form (number of hidden layers, number of units per layer)
    learning_rates_to_try = [0.001, 0.01, 0.1]
    epochs_to_try = [100, 500, 1000, 2000]

    #initialize vairables to store the information for the best xor training
    xor_best_loss = 1.0
    xor_best_hyperparameters = tuple()
    xor_best_epochs = 0.0
    xor_best_split = 0.0
    xor_best_lr = 0.0

    #loop through all permutations of the parameters
    for split in splits_to_try:
            for hyperparameters in hyperparameters_to_try:
                for lr in learning_rates_to_try:
                    for epochs in epochs_to_try:

                        #train
                        loss = train_and_test('xor', hyperparameters, lr, epochs, split)
                        print_train_status('xor', hyperparameters[0], hyperparameters[1], lr, epochs, split, loss)

                        #if the loss was the new lowest
                        if(loss < xor_best_loss):

                            #save off all the parameters
                            xor_best_loss = loss
                            xor_best_hyperparameters = hyperparameters
                            xor_best_lr = lr
                            xor_best_epochs = epochs
                            xor_best_split = split

    #initialize vairables to store the information for the best adder training
    adder_best_loss = 1.0
    adder_best_hyperparameters = tuple()
    adder_best_epochs = 0.0
    adder_best_split = 0.0
    adder_best_lr = 0.0

    #loop through all permutations of the parameters
    for split in splits_to_try:
        for hyperparameters in hyperparameters_to_try:
            for lr in learning_rates_to_try:
                for epochs in epochs_to_try:

                    #train
                    loss = train_and_test('adder', hyperparameters, lr, epochs, split)
                    print_train_status('adder', hyperparameters[0], hyperparameters[1], lr, epochs, split, loss)

                    #if the loss was the new lowest
                    if(loss < adder_best_loss):

                        #save off all the parameters
                        adder_best_loss = loss
                        adder_best_hyperparameters = hyperparameters
                        adder_best_lr = lr
                        adder_best_epochs = epochs
                        adder_best_split = split

    #log all the results
    print("\n------------------------------------------------------------------------------------------------------")
    print("\nThe best network I tested for the xor dataset has the following details:")
    print("hidden layers:", xor_best_hyperparameters[0])
    print("units per hidden layer:", xor_best_hyperparameters[1])
    print("learning rate:", xor_best_lr)
    print("epochs:", xor_best_epochs)
    print("train/test split:", xor_best_split)
    print("testloss:", xor_best_loss, '\n')
    print("------------------------------------------------------------------------------------------------------")

    print("\nThe best network I tested for the adder dataset has the following details:")
    print("hidden layers:", adder_best_hyperparameters[0])
    print("units per hidden layer:", adder_best_hyperparameters[1])
    print("learning rate:", adder_best_lr)
    print("epochs:", adder_best_epochs)
    print("train/test split:", adder_best_split)
    print("testloss:", adder_best_loss, '\n')
    print("\n------------------------------------------------------------------------------------------------------")

    t1 = time.time()
    print("\nTotal runtime: ", int((t1 - t0) // 60), "m", round((t1 - t0) % 60, 1), "s")


if __name__ == "__main__":
    main()