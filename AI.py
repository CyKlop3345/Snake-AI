from constants import *

from snake import Snake
from apple import Apple


class AI:
    def __init__(self):
        self.apples = []
        # Neurons
        self.neuronCount_in = 8 # 4 -- danger detector
                                # 4 -- apple direction
        self.neuronCount_hid1 = 6
        self.neuronCount_hid2 = 6
        self.neuronCount_out = 3 # next direction (local (turn left or right or go forward))

        self.layer_input = [0 for i in range(self.neuronCount_in)]
        self.layer_hidden_1 = [0 for i in range(self.neuronCount_hid1)]
        self.layer_hidden_2 = [0 for i in range(self.neuronCount_hid2)]
        self.layer_output = [0 for i in range(self.neuronCount_out)]

        # Connections (matrix)
        # num of the conection is force of influence to the neuron of the next layer
        Connections_in_hid1 = [[0.5 for i in range(self.neuronCount_in)] for i in range(self.neuronCount_hid1)]
        Connections_hid1_hid2 = [[0.5 for i in range(self.neuronCount_hid1)] for i in range(self.neuronCount_hid2)]
        Connections_hid2_out = [[0.5 for i in range(self.neuronCount_hid2)] for i in range(self.neuronCount_out)]


    def run(self):
        calculate_Layer_input()


    def calculate_Layer_input(self):
        snakePos = self.snake.getPos()
        # Reset input layer
        for i in range(self.neuronCount_in):
            self.layer_input[i] = 0
        # Check for dangerous (self segments or boundary)
        for offset in range(5, 0, -1):
            for snakeSegment in self.snake.pos:
                # Left
                if snakePos[0] - offset == snakeSegment or snakePos[0] - offset < 0:
                    self.layer_input[0] = 1 - 0.2*offset
                # Right
                elif snakePos[0] + offset == snakeSegment or snakePos[0] + offset < GRID_RES[0]-1:
                    self.layer_input[1] = 1 - 0.2*offset
                # Up
                if snakePos[1] - offset == snakeSegment or snakePos[1] - offset < 0:
                    self.layer_input[2] = 1 - 0.2*offset
                # Down
                elif snakePos[1] + offset == snakeSegment or snakePos[1] + offset < GRID_RES[1]-1:
                    self.layer_input[3] = 1 - 0.2*offset
        # Apple finding
        # finding nearest apple
        apple_nearest
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
    # sigmoid function return number from 0 to 1
        return 1 / (1 + (E_constant ** x))
        # x == 0    => y = 0.5
        # x -> inf  => y -> 1
        # x -> -inf => y -> 0


    def setSnakeControl(self, snake):
        self.snake = snake


    def addApple(self, apple):
        self.apples.append(apple)


    def draw(self):
        pass


if __name__ == "__main__":
    ai = AI()
    ai.setSnakeControl(Snake(0))
    ai.addApple(Apple(0))
    ai.run()
