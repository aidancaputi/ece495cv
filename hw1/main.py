from datasets import *

def main():

    #create a dict where keys are 'x', 'y', and 'out'
    xor_data = xor_dataset()
    print(xor_data)

    

if __name__ == "__main__":
    main()