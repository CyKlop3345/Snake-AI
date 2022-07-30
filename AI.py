from constants import *


class AI:
    def __init__(self):
        # Neurons
        neuronCount_in = 7  # 3 -- danger detector ( forward, left, right (local) )
                            # ( backward is not needed )
                            # 4 -- apple direction (local)
        neuronCount_hid1 = 6
        neuronCount_hid2 = 6
        neuronCount_out = 3 # next direction (local (turn left or right or go forward))

        self.layer_input = [0 for i in range(neuronCount_in)]
        self.layer_hidden_1 = [0 for i in range(neuronCount_hid1)]
        self.layer_hidden_2 = [0 for i in range(neuronCount_hid2)]
        self.layer_output = [0 for i in range(neuronCount_out)]

        # Connections (matrix)
        # num of the conection is force of influence to the neuron of the next layer
        Connections_in_hid1 = [[0.5 for i in range(neuronCount_in)] for i in range(neuronCount_hid1)]
        Connections_hid1_hid2 = [[0.5 for i in range(neuronCount_hid1)] for i in range(neuronCount_hid2)]
        Connections_hid2_out = [[0.5 for i in range(neuronCount_hid2)] for i in range(neuronCount_out)]


    def fname(self):
        pass


    def sigmoid(self, x):
    # sigmoid function return number from 0 to 1
        return 1 / (1 + (E_constant ** x))
        # x == 0    => y = 0.5
        # x -> inf  => y -> 1
        # x -> -inf => y -> 0


    def setSnakeControl(self, snake):
        self.snake = snake


    def draw(self):
        pass


if __name__ == "__main__":
    ai = AI()
