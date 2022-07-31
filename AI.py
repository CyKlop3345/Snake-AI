from constants import *
import random
from snake import Snake
from apple import Apple


class AI:
    def __init__(self):
        random.seed()

        self.apples = []
        # Neurons count
        self.neuronCount_in = 8 # 4 -- danger detector
                                # 4 -- apple direction
        self.neuronCount_hid1 = 6
        self.neuronCount_hid2 = 6
        self.neuronCount_out = 4 # next direction (local (turn left or right or go forward))

        # Neurons vector (1-dim matrix)
        self.layer_input = [[0 for i in range(self.neuronCount_in)]]
        self.layer_hidden_1 = [[0 for i in range(self.neuronCount_hid1)]]
        self.layer_hidden_2 = [[0 for i in range(self.neuronCount_hid2)]]
        self.layer_output = [[0 for i in range(self.neuronCount_out)]]

        # Connections (matrix)
        # num of the conection is force of influence to the neuron of the next layer
        self.connections_in_hid1 = [[random.random() for j in range(self.neuronCount_hid1)] for i in range(self.neuronCount_in)]
        self.connections_hid1_hid2 = [[random.random() for j in range(self.neuronCount_hid2)] for i in range(self.neuronCount_hid1)]
        self.connections_hid2_out = [[random.random() for j in range(self.neuronCount_out)] for i in range(self.neuronCount_hid2)]


    def run(self):
        self.calculate_Layer_input()
        self.layer_hidden_1 = self.matrixMultipl(self.layer_input, self.connections_in_hid1)
        self.sigmoid(self.layer_hidden_1)
        self.layer_hidden_2 = self.matrixMultipl(self.layer_hidden_1, self.connections_hid1_hid2)
        self.sigmoid(self.layer_hidden_2)
        self.layer_output = self.matrixMultipl(self.layer_hidden_2, self.connections_hid2_out)
        self.sigmoid(self.layer_output)

        choiceDirection = -1
        choiceWeight = 0
        for i in range(len(self.layer_output[0])):
            if self.layer_output[0][i] > choiceWeight:
                choiceInd = i
                choiceWeight = self.layer_output[0][i]

        self.snake.setDirection(choiceInd)
        self.snake.moveForward()



    def calculate_Layer_input(self):
        snakePos = self.snake.getPos()
        # Reset input layer
        for i in range(self.neuronCount_in):
            self.layer_input[0][i] = 0
        # Check for dangerous (self segments or boundary)
        for offset in range(5, 0, -1):
            # Self segments
            for snakeSegment in self.snake.pos:
                # Left
                if [snakePos[0] - offset, snakePos[1]] == snakeSegment:
                    self.layer_input[0][0] = max(self.layer_input[0][0], 6 - offset)
                # Right
                if [snakePos[0] + offset, snakePos[1]] == snakeSegment:
                    self.layer_input[0][1] = max(self.layer_input[0][1], 6 - offset)
                # Up
                if [snakePos[0], snakePos[1] - offset] == snakeSegment:
                    self.layer_input[0][2] = max(self.layer_input[0][2], 6 - offset)
                # Down
                if [snakePos[0], snakePos[1] + offset] == snakeSegment:
                    self.layer_input[0][3] = max(self.layer_input[0][3], 6 - offset)

            # Boundary
            # Left
            if snakePos[0] - offset < 0:
                self.layer_input[0][0] = max(self.layer_input[0][0], 6 - offset)
            # Right
            if snakePos[0] + offset > GRID_RES[0]-1:
                self.layer_input[0][1] = max(self.layer_input[0][1], 6 - offset)
            # Up
            if snakePos[1] - offset < 0:
                self.layer_input[0][2] = max(self.layer_input[0][2], 6 - offset)
            # Down
            if snakePos[1] + offset > GRID_RES[1]-1:
                self.layer_input[0][3] = max(self.layer_input[0][3], 6 - offset)

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
            self.layer_input[0][4] = 1
        # Right
        elif apple_nearest.getPos()[0] - snakePos[0] > 0:
            self.layer_input[0][5] = 1
        # Up
        if apple_nearest.getPos()[1] - snakePos[1] < 0:
            self.layer_input[0][6] = 1
        # Down
        if apple_nearest.getPos()[1] - snakePos[1] > 0:
            self.layer_input[0][5] = 1


    # Testing function (maybe numpy is the better way)
    def matrixMultipl(self, matrix_A, matrix_B):
        columns_A = len(matrix_A[0])
        rows_B = len(matrix_B)
        # columns_A and rows_B must be equal
        if columns_A != rows_B:
            print("Wrong matrixes")
            return [-1]

        rows_AC = len(matrix_A)
        columns_BC = len(matrix_B[0])
        matrix_C = [[0 for j in range(columns_BC)] for i in range(rows_AC)]
        for i in range(rows_AC):
            for j in range(columns_BC):
                for k in range(columns_A):
                    matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j]
        return matrix_C


    def sigmoid(self, x):
    # sigmoid function return number from 0 to 1
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] = 1 / (1 + (E_constant ** x[i][j]))
        return x
        # return 1 / (1 + (E_constant ** x))
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
