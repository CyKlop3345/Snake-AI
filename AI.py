from constants import *
import numpy

from snake import Snake
from apple import Apple


class AI:
    def __init__(self):
        self.apples = []
        # Neurons count
        self.neuronCount_in = 8 # 4 -- danger detector
                                # 4 -- apple direction
        self.neuronCount_hid1 = 6
        self.neuronCount_hid2 = 6
        self.neuronCount_out = 4 # next direction (local (turn left or right or go forward))

        # Neurons vector (1-dim matrix)
        self.layer_input = numpy.zeros(self.neuronCount_in, dtype=numpy.int16)
        self.layer_hidden_1 = numpy.zeros(self.neuronCount_hid1)
        self.layer_hidden_2 = numpy.zeros(self.neuronCount_hid2)
        self.layer_output = numpy.zeros(self.neuronCount_out)

        # Connections (matrix)
        # num of the conection is force of influence to the neuron of the next layer
        self.connections_in_hid1 = numpy.random.random_sample((self.neuronCount_in, self.neuronCount_hid1))
        self.connections_hid1_hid2 = numpy.random.random_sample((self.neuronCount_hid1, self.neuronCount_hid2))
        self.connections_hid2_out = numpy.random.random_sample((self.neuronCount_hid2, self.neuronCount_out))


    def run(self):
        self.calculate_Layer_input()
        self.layer_hidden_1 = numpy.dot(self.layer_input, self.connections_in_hid1)
        self.sigmoid(self.layer_hidden_1)
        self.layer_hidden_2 = numpy.dot(self.layer_hidden_1, self.connections_hid1_hid2)
        self.sigmoid(self.layer_hidden_2)
        self.layer_output = numpy.dot(self.layer_hidden_2, self.connections_hid2_out)
        self.sigmoid(self.layer_output)

        choiceIndex = numpy.argmax(self.layer_output)

        self.snake.setDirection(choiceIndex)
        self.snake.moveForward()


    def calculate_Layer_input(self):
        snakePos = self.snake.getHeadPos()
        # Reset input layer
        for i in range(self.neuronCount_in):
            self.layer_input[i] = 0
        # Check for dangerous (self segments or boundary)
        for offset in range(5, 0, -1):
            # Self segments
            for snakeSegment in self.snake.getSegmPos():
                # Left
                if [snakePos[0] - offset, snakePos[1]] == snakeSegment:
                    self.layer_input[0] = max(self.layer_input[0], 6 - offset)
                # Right
                elif [snakePos[0] + offset, snakePos[1]] == snakeSegment:
                    self.layer_input[1] = max(self.layer_input[1], 6 - offset)
                # Up
                elif [snakePos[0], snakePos[1] - offset] == snakeSegment:
                    self.layer_input[2] = max(self.layer_input[2], 6 - offset)
                # Down
                elif [snakePos[0], snakePos[1] + offset] == snakeSegment:
                    self.layer_input[3] = max(self.layer_input[3], 6 - offset)

            # Boundary
            # Left
            if snakePos[0] - offset < 0:
                self.layer_input[0] = max(self.layer_input[0], 6 - offset)
            # Right
            if snakePos[0] + offset > GRID_RES[0]-1:
                self.layer_input[1] = max(self.layer_input[1], 6 - offset)
            # Up
            if snakePos[1] - offset < 0:
                self.layer_input[2] = max(self.layer_input[2], 6 - offset)
            # Down
            if snakePos[1] + offset > GRID_RES[1]-1:
                self.layer_input[3] = max(self.layer_input[3], 6 - offset)

        # Apple finding
        # finding nearest apple
        # apple_nearest
        apple_nearest_distance = 100
        for apple in self.apples:
            apple_distance = abs(snakePos[0] - apple.getPos()[0]) + abs(snakePos[1] - apple.getPos()[1])
            if apple_distance < apple_nearest_distance:
                apple_distance = apple_nearest_distance
                apple_nearest = apple
        # Identification the right side
        # Left
        if apple_nearest.getPos()[0] - snakePos[0] < 0:
            self.layer_input[4] = 1
        # Right
        elif apple_nearest.getPos()[0] - snakePos[0] > 0:
            self.layer_input[5] = 1
        # Up
        if apple_nearest.getPos()[1] - snakePos[1] < 0:
            self.layer_input[6] = 1
        # Down
        if apple_nearest.getPos()[1] - snakePos[1] > 0:
            self.layer_input[5] = 1


    def sigmoid(self, x):
    # Sigmoid function return number from 0 to 1
    # For 1-dim arrays
        for i in range(len(x)):
            x[i] = 1 / (1 + (E_constant ** x[i]))
        return x
    # For integers
        # return 1 / (1 + (E_constant ** x))
    ''' Hint
    x == 0    => y = 0.5
    x -> inf  => y -> 1
    x -> -inf => y -> 0
    '''


    def setSnakeControl(self, snake):
        self.snake = snake


    def addApple(self, apple):
        self.apples.append(apple)


    def draw(self):
        pass


if __name__ == "__main__":
    ai = AI()
    # Blank. Can't use show() functions
    ai.setSnakeControl(Snake(0))
    ai.addApple(Apple(0))
    ai.run()
