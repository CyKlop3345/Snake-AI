from constants import *

import numpy as np
import matplotlib.pyplot as plt
import random
import pygame
from pathlib import Path

from snake import Snake
from apple import Apple


# Mistakes calculation
def linear_loss(y, y_right):
    return np.sum(np.absolute(y-y_right))

def quadratic_loss(y, y_right):
    return np.sum(np.power(y-y_right, 2))

def cross_entropy_loss(y, y_right):
    return -np.sum(y_right * np.log(y))

def softmax(x):
    # Activation func for output layers
    # Converts a numeric array to фт array of probabilities
    out = np.exp(x)
    sum = np.sum(out)
    return out / sum



# Activation funcs and derivatives
def sigmoid(x):
    ''' Hint
    x == 0    => y = 0.5
    x -> inf  => y -> 1
    x -> -inf => y -> 0
    '''
    out = np.copy(x)
    out = 1 / ( 1 + np.exp(-out) )
    return out

def sigmoid_deriv(x):
    out = np.copy(x)
    out = sigmoid(out) * (1-sigmoid(out))
    return out

def relu(x):
    out = np.copy(x)
    out[out < 0] = 0
    out[out > 1] = 1
    return out

def relu_deriv(x):
    out = np.copy(x)
    out[out < 0] = 0
    out[out > 1] = 0
    out[(out >= 0) & (out <= 1)] = 1
    return out

def leaky_relu(x):
    out = np.copy(x)
    out[out < 0] *= 0.01
    out[out > 1] = 1 + 0.01*(out[out > 1]-1)
    return out

def leaky_relu_deriv(x):
    out = np.copy(x)
    out[out < 0] = 0.01
    out[out > 1] = 0.01
    out[(out >= 0) & (out <= 1)] = 1
    return out

# Select activation function (sigmoid or relu)
activ_func = leaky_relu
activ_func_deriv = leaky_relu_deriv



class AI:
    # Initialization
    def __init__(self, surface=None):

        # surface for drawing
        self.surface = surface

        # Numpy settings
        np.set_printoptions(precision=2, suppress=True) # numpy print settings


        # Training coefficient (CONST)
        self.ALPHA = 0.001


        # Mistakes (plotting)
        # Different types of mistakes analysis
        # corteg -- the average mistake value fpr the training array
        # plot 1: linear loss
        self.mists_lin = np.array([])
        self.mists_lin_corteg = np.array([])
        # plot 2: Square loss
        self.mists_sq = np.array([])
        self.mists_sq_corteg = np.array([])
        # plot 3: Cross Entropy loss
        self.mists_entr = np.array([])
        self.mists_entr_corteg = np.array([])


        # Neurons count (CONST)
        self.C_IN = 5   # 3 -- danger detector  (left, forward, right)
                        # 2 -- apple direction  (forward_or_backwrd, left_or_right)
        self.C_H1 = 4
        self.C_H2 = 4
        self.C_OUT = 3  # 3 next direction (turning left, right or step forward)


        # Layers of Neurons (vector)
        self.l_in = np.zeros((1, self.C_IN))
        self.l_out_right = np.zeros((1, self.C_OUT)) # For training

        self.l_h1 = np.zeros((1, self.C_H1))
        self.l_h2 = np.zeros((1, self.C_H2))
        self.l_out = np.zeros((1, self.C_OUT))
        self.choice = -1


        # Weights (matrix)
        self.W_in_h1 = None
        self.W_h1_h2 = None
        self.W_h2_out = None


        # Shift (vector)
        self.s_in_h1 = None
        self.s_h1_h2 = None
        self.s_h2_out = None


        # File to collect arrays (with marks about layers size)
        self.arrays_filename = f"Arrays_{self.C_IN}_{self.C_H1}_{self.C_H2}_{self.C_OUT}"
        self.arrays_dir = Path.cwd() / 'Arrays_Data'
        self.arrays_full_path = self.arrays_dir / self.arrays_filename
        # create folders
        if not self.arrays_dir.is_dir():
            self.arrays_dir.mkdir()


        # Check file exists
        if (self.arrays_full_path.with_suffix(".npz")).is_file():
            # load weights and shifts data from the file
            self.load_data()
        else:
            # Randomization weights and shifts
            self.W_in_h1 = np.random.uniform(-1, 1, (self.C_IN, self.C_H1))
            self.W_h1_h2 = np.random.uniform(-1, 1, (self.C_H1, self.C_H2))
            self.W_h2_out = np.random.uniform(-1, 1, (self.C_H2, self.C_OUT))

            self.s_in_h1 = np.random.uniform(-1, 1, (1, self.C_H1))
            self.s_h1_h2 = np.random.uniform(-1, 1, (1, self.C_H2))
            self.s_h2_out = np.random.uniform(-1, 1, (1, self.C_OUT))


        # Mistakes
        self.m_h1 = np.zeros((1, self.C_H1))
        self.m_h2 = np.zeros((1, self.C_H2))
        self.m_out = np.zeros((1, self.C_OUT))



    # Public funcs
    def run(self, _in):
        # Use AI for choosing next step
        # send input data

        # Setting input layer
        self.l_in = np.array(_in, dtype=float).reshape(1,len(_in))
        # Calculate output
        self.forward()
        # Choosing commad based on probability
        # self.choice = np.random.choice([-1,0,1], p=self.l_out[0])

        # Choosing command based on max value
        # convert [0, 1, 2] array into [-1, 0, 1]
        self.choice = np.argmax(self.l_out)-1

    def training(self, _in, _out):
        # AI training
        # using input and correct output data
        # (the correct output data is knows)

        # setting input and correct output layers
        self.l_in = np.array(_in, dtype=float).reshape(1,len(_in))
        self.l_out_right = np.array(_out).reshape(1,len(_out))
        # Calculate output, finding mistakes and correction matrixes
        self.forward()
        self.choice = np.argmax(self.l_out)
        self.backward()
        self.update()


    # Private funcs
    def forward(self):
        # Finding output
        self.l_h1_row = self.l_in @ self.W_in_h1 + self.s_in_h1
        self.l_h1 = activ_func(self.l_h1_row)
        self.l_h2_row = self.l_h1_row @ self.W_h1_h2 + self.s_h1_h2
        self.l_h2 = activ_func(self.l_h2_row)
        self.l_out_row = self.l_h2_row @ self.W_h2_out + self.s_h2_out
        self.l_out = softmax(self.l_out_row)

    def backward(self):
        # Finding mistakes
        self.m_out = self.l_out - self.l_out_right
        self.m_h2 = self.m_out @ self.W_h2_out.T * activ_func_deriv(self.l_h2)
        self.m_h1 = self.m_h2 @ self.W_h1_h2.T * activ_func_deriv(self.l_h1)

    def update(self):
        # Updating weights and shiftings
        self.W_in_h1 -= self.ALPHA * self.l_in.T @ self.m_h1
        self.s_in_h1 -= self.ALPHA * np.sum(self.m_h1, axis=0, keepdims=True)
        self.W_h1_h2 -= self.ALPHA * self.l_h1_row.T @ self.m_h2
        self.s_h1_h2 -= self.ALPHA * np.sum(self.m_h2, axis=0, keepdims=True)
        self.W_h2_out -= self.ALPHA * self.l_h2_row.T @ self.m_out
        self.s_h2_out -= self.ALPHA * np.sum(self.m_out, axis=0, keepdims=True)


    # Getters
    def get_choice(self):
        # Get choice from the other source
        return self.choice

    def get_output(self):
        # Get output layer values
        # after softmax function
        return self.l_out


    # Mistakes calculation with different modification
    def calc_mist_lin(self):
        mist = linear_loss(self.l_out, self.l_out_right)
        self.mists_lin = np.append(self.mists_lin, mist)

    def calc_mist_sq(self):
        mist = quadratic_loss(self.l_out, self.l_out_right)
        self.mists_sq = np.append(self.mists_sq, mist)

    def calc_mists_entr(self):
        # Works normally only with classification tasks
        # when in the correct output there is only one "1"
        # without less numbers, like 0.5, 0.33 etc.
        # [0, ..., 0, 1, 0, ..., 0]
        E = cross_entropy_loss(self.l_out, self.l_out_right)
        self.mists_entr = np.append(self.mists_entr, E)

    def calc_mist_corteg(self, last_count):
        # corteg -- the average mistake value for the training array
        mist_lin = 0
        mist_sq = 0
        mist_entr = 0
        for i in range(last_count):
            mist_lin += self.mists_lin[-i-1]
            mist_sq += self.mists_sq[-i-1]
            mist_entr += self.mists_entr[-i-1]
        mist_lin /= last_count
        mist_sq /= last_count
        mist_entr /= last_count
        self.mists_lin_corteg = np.append(self.mists_lin_corteg, mist_lin)
        self.mists_sq_corteg = np.append(self.mists_sq_corteg, mist_sq)
        self.mists_entr_corteg = np.append(self.mists_entr_corteg, mist_entr)


    # Debugging data
    def show_graphics(self):
            # Graphics settings
            figure, axes = plt.subplots(2, 2, figsize=(8.96, 6.72))
            axes[0,0].set_title('Linear')
            axes[0,1].set_title('Square')
            axes[1,0].set_title('Cross-entropy')
            axes[0,0].set_ylim([0, 1])
            axes[0,1].set_ylim([0, 1])
            axes[1,0].set_ylim([0, 1])
            axes[1,1].set_ylim([0, 1])

            # Check for the empty arrays
            if self.mists_sq_corteg.size == 0:
                return

            # First graphic (linear analysis)
            axes[0,0].plot(self.mists_lin_corteg, 'r.', markeredgewidth = 0)

            axes[0,0].text(0.02, 0.94, f"start={'%.3f' % self.mists_lin_corteg[0]}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,0].text(0.5, 0.94, f"end={'%.3f' % self.mists_lin_corteg[-1]}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,0].text(0.02, 0.78, f"min={'%.3f' % self.mists_lin_corteg.min()}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,0].text(0.5, 0.78, f"max={'%.3f' % self.mists_lin_corteg.max()}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})

            # Second graphic (square analysis)
            axes[0,1].plot(self.mists_sq_corteg, 'b.', markeredgewidth = 0)

            axes[0,1].text(0.02, 0.94, f"start={'%.3f' % self.mists_sq_corteg[0]}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,1].text(0.5, 0.94, f"end={'%.3f' % self.mists_sq_corteg[-1]}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,1].text(0.02, 0.78, f"min={'%.3f' % self.mists_sq_corteg.min()}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,1].text(0.5, 0.78, f"max={'%.3f' % self.mists_sq_corteg.max()}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})

            # Third graphics (entropy analysis)
            axes[1,0].plot(self.mists_entr_corteg, 'y.', markeredgewidth = 0)

            axes[1,0].text(0.02, 0.94, f"start={'%.3f' % self.mists_entr_corteg[0]}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[1,0].text(0.5, 0.94, f"end={'%.3f' % self.mists_entr_corteg[-1]}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[1,0].text(0.02, 0.78, f"min={'%.3f' % self.mists_entr_corteg.min()}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[1,0].text(0.5, 0.78, f"max={'%.3f' % self.mists_entr_corteg.max()}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})

            # Showing
            plt.show()

    def show_data(self):
        # Print weights and shifts into a consol
        print("\nW_in_h1\n", self.W_in_h1)
        print("\nW_h1_h2\n", self.W_h1_h2)
        print("\nW_h2_out\n", self.W_h2_out)
        print("\ns_in_h1\n", self.s_in_h1)
        print("\ns_h1_h2\n", self.s_h1_h2)
        print("\ns_h2_out\n", self.s_h2_out)


    # Data manipulation
    def save_data(self):
        print("!!!!!")
        # Save matrixes into the file
        np.savez(self.arrays_full_path,
                    self.W_in_h1, self.W_h1_h2, self.W_h2_out,
                    self.s_in_h1, self.s_h1_h2, self.s_h2_out)

    def load_data(self):
        # load matrixes from the file
        file = np.load(self.arrays_full_path.with_suffix(".npz"))
        self.W_in_h1 = file['arr_0']
        self.W_h1_h2 = file['arr_1']
        self.W_h2_out = file['arr_2']
        self.s_in_h1 = file['arr_3']
        self.s_h1_h2 = file['arr_4']
        self.s_h2_out = file['arr_5']


    # AI visualization
    def set_draw_property(self):
        # Set property for the networ visualization
        layers_sizes = [self.C_IN, self.C_H1, self.C_H2, self.C_OUT]
        layer_size_large = max(layers_sizes)

        offset = [140, 60]              # offset between nodes (x, y)
        offset_start = [40,40]          # offset between first node and left-up corner (x, y)

        offset_layer = []               # centering of the layers
        for size in layers_sizes:
            offset_layer.append((layer_size_large - size) * 30)

        self.node_pos = np.array((  np.zeros((self.C_IN, 2), dtype=int),
                                    np.zeros((self.C_H1, 2), dtype=int),
                                    np.zeros((self.C_H2, 2), dtype=int),
                                    np.zeros((self.C_OUT, 2), dtype=int)  ))

        '''
        self.node_pos[i]        -- layer (in, h1, h2, out)
        self.node_pos[i][j]     -- node coordinates
        self.node_pos[i][j,0]  -- node x-coordinate
        self.node_pos[i][j,1]  -- node y-coordinate
        '''
        # in layer
        self.node_pos[0][:,0] = offset_start[0]
        self.node_pos[0][:,1] = np.arange(self.C_IN) * offset[1] + offset_start[1] + offset_layer[0]
        # h1 layer
        self.node_pos[1][:,0] = offset[0] + offset_start[0]
        self.node_pos[1][:,1] = np.arange(self.C_H1) * offset[1] + offset_start[1] + offset_layer[1]
        # h2 layer
        self.node_pos[2][:,0] = offset[0] * 2 + offset_start[0]
        self.node_pos[2][:,1] = np.arange(self.C_H2) * offset[1] + offset_start[1] + offset_layer[2]
        # out layer
        self.node_pos[3][:,0] = offset[0] * 3 + offset_start[0]
        self.node_pos[3][:,1] = np.arange(self.C_OUT) * offset[1] + offset_start[1] + offset_layer[3]

        self.node_size = 20 # Circle size

        # Drawing connections
        weights = [sigmoid(self.W_in_h1), sigmoid(self.W_h1_h2), sigmoid(self.W_h2_out)]
        for layer in range(3):                          # cont of layers minus 1
            for node_1 in range(layers_sizes[layer]):         # "from" layer
                for node_2 in range(layers_sizes[layer+1]):   # "to" layer
                    # Choosing a  medium color
                    value = weights[layer][node_1][node_2]
                    color = CL_conect_active * value + CL_conect_inactive * (1-value)
                    pygame.draw.line(self.surface, color,
                                     self.node_pos[layer][node_1], self.node_pos[layer+1][node_2], 2)

    def draw_update(self):
        # Updating
        # Drawing nodes
        layers_sizes = [self.C_IN, self.C_H1, self.C_H2, self.C_OUT]
        layers = [np.array(self.l_in, copy=True), relu(self.l_h1), relu(self.l_h2), relu(self.l_out)]
        # convert [-1, 0, 1] into [0, 0.5, 1]
        # for lasts 2 value (apple directions)
        layers[0][0,3:] = (layers[0][0,3:]+1)/2
        for layer in range(4):
            for node in range(layers_sizes[layer]):
                # Choosing a  medium color
                value = layers[layer][0,node]
                color = CL_node_active * value + CL_node_inactive * (1-value)
                pygame.draw.circle(self.surface, color,
                                    self.node_pos[layer][node], self.node_size)


# Start point
if __name__ == "__main__":

    ai = AI()

                    # barrier|apple | choice       (l -- left, f -- forward, r -- right)
                    #  l f r bf lr    l     f     r
    train_array = [ ( (0,0,0, 0, 0), (0.33, 0.33, 0.33) ), # Good

                    ( (0,0,0, 1, 0), (0,    1,    0   ) ), # Good
                    ( (0,0,0, 1, 1), (0,    0.5,  0.5 ) ), # Good
                    ( (0,0,0, 0, 1), (0,    0,    1   ) ), # Good
                    ( (0,0,0,-1, 1), (0,    0,    1   ) ), # Good
                    ( (0,0,0,-1, 0), (0.5,  0,    0.5 ) ), # Good
                    ( (0,0,0,-1,-1), (1,    0,    0   ) ), # Good
                    ( (0,0,0, 0,-1), (1,    0,    0   ) ), # Good
                    ( (0,0,0, 1,-1), (0.5,  0.5,  0   ) ), # Good
                    #  l f r bf lr    l     f     r
                    ( (0,1,0, 0, 0), (0.5,  0,    0.5 ) ), # Good

                    ( (0,1,0, 1, 0), (0.5,  0,    0.5 ) ), # Good
                    ( (0,1,0, 1, 1), (0,    0,    1   ) ), # Good
                    ( (0,1,0, 0, 1), (0,    0,    1   ) ), # Good
                    ( (0,1,0,-1, 1), (0,    0,    1   ) ), # Good
                    ( (0,1,0,-1, 0), (0.5,  0,    0.5 ) ), # Good
                    ( (0,1,0,-1,-1), (1,    0,    0   ) ), # Good
                    ( (0,1,0, 0,-1), (1,    0,    0   ) ), # Good
                    ( (0,1,0, 1,-1), (1,    0,    0   ) ), # Good
                    #  l f r bf lr    l     f     r
                    ( (1,0,0, 0, 0), (0,    0.5,  0.5 ) ), # Good

                    ( (1,0,0, 1, 0), (0,    1,    0   ) ), # Good
                    ( (1,0,0, 1, 1), (0,    0.5,  0.5 ) ), # Good
                    ( (1,0,0, 0, 1), (0,    0,    1   ) ), # Good
                    ( (1,0,0,-1, 1), (0,    0,    1   ) ), # Good
                    ( (1,0,0,-1, 0), (0,    0,    1   ) ), # Good
                    ( (1,0,0,-1,-1), (0,    1,    0   ) ), # Good
                    ( (1,0,0, 0,-1), (0,    1,    0   ) ), # Good
                    ( (1,0,0, 1,-1), (0,    1,    0   ) ), # Good
                    #  l f r bf lr    l     f     r
                    ( (0,0,1, 0, 0), (0.5,  0.5,  0   ) ), # Good

                    ( (0,0,1, 1, 0), (0,    1,    0   ) ), # Good
                    ( (0,0,1, 1, 1), (0,    1,    0   ) ), # Good
                    ( (0,0,1, 0, 1), (0,    1,    0   ) ), # Good
                    ( (0,0,1,-1, 1), (0,    1,    0   ) ), # Good
                    ( (0,0,1,-1, 0), (1,    0,    0   ) ), # Good
                    ( (0,0,1,-1,-1), (1,    0,    0   ) ), # Good
                    ( (0,0,1, 0,-1), (1,    0,    0   ) ), # Good
                    ( (0,0,1, 1,-1), (0.5,  0.5,  0   ) ), # Good
                    #  l f r bf lr    l     f     r
                    ( (1,1,0, 0, 0), (0,    0,    1) ), # Good

                    ( (1,1,0, 1, 0), (0,    0,    1   ) ), # Good
                    ( (1,1,0, 1, 1), (0,    0,    1   ) ), # Good
                    ( (1,1,0, 0, 1), (0,    0,    1   ) ), # Good
                    ( (1,1,0,-1, 1), (0,    0,    1   ) ), # Good
                    ( (1,1,0,-1, 0), (0,    0,    1   ) ), # Good
                    ( (1,1,0,-1,-1), (0,    0,    1   ) ), # Good
                    ( (1,1,0, 0,-1), (0,    0,    1   ) ), # Good
                    ( (1,1,0, 1,-1), (0,    0,    1   ) ), # Good
                    #  l f r bf lr    l     f     r
                    ( (0,1,1, 0, 0), (1,    0,    0   ) ), # Good

                    ( (0,1,1, 1, 0), (1,    0,    0   ) ), # Good
                    ( (0,1,1, 1, 1), (1,    0,    0   ) ), # Good
                    ( (0,1,1, 0, 1), (1,    0,    0   ) ), # Good
                    ( (0,1,1,-1, 1), (1,    0,    0   ) ), # Good
                    ( (0,1,1,-1, 0), (1,    0,    0   ) ), # Good
                    ( (0,1,1,-1,-1), (1,    0,    0   ) ), # Good
                    ( (0,1,1, 0,-1), (1,    0,    0   ) ), # Good
                    ( (0,1,1, 1,-1), (1,    0,    0   ) ), # Good
                    #  l f r bf lr    l     f     r
                    ( (1,0,1, 0, 0), (0,    1,    0   ) ), # Good

                    ( (1,0,1, 1, 0), (0,    1,    0   ) ), # Good
                    ( (1,0,1, 1, 1), (0,    1,    0   ) ), # Good
                    ( (1,0,1, 0, 1), (0,    1,    0   ) ), # Good
                    ( (1,0,1,-1, 1), (0,    1,    0   ) ), # Good
                    ( (1,0,1,-1, 0), (0,    1,    0   ) ), # Good
                    ( (1,0,1,-1,-1), (0,    1,    0   ) ), # Good
                    ( (1,0,1, 0,-1), (0,    1,    0   ) ), # Good
                    ( (1,0,1, 1,-1), (0,    1,    0   ) )  # Good
                    ]

    # Testing
    '''
    for set in train_array:
        ai.run(set[0])

        # Notification about wrong choice
        if set[1][ai.get_choice()+1] == 0:
            print("!!!!!          There is a fall                   !!!!!")

        # Calc mistakes
        lin = linear_loss(ai.get_output(), set[1])
        sq = quadratic_loss(ai.get_output(), set[1])
        E = cross_entropy_loss(ai.get_output(), set[1])

        print(  f"input: {set[0]}\n"
                f"output: {set[1]}\n"
                f"prob: {ai.get_output()}\n"
                f"choice: {ai.get_choice()}\n"
                f"lin {lin :.2f}\n"
                f"sq: {sq :.2f}\n"
                f"E: {E :.2f}\n"
                )

    '''

    # Training

    iterat_num = 1500
    iterat_step = 1000  # percentage accuracy
    # To the good training needs from 1.5k to several k-s
    # Lasts 10 sec per 1k iterations
    for i in range(iterat_num):
        random.shuffle(train_array) # mixing training array

        for set in train_array:
            ai.training(set[0], set[1])
            # Plotting (mists)
            if i % (iterat_num/iterat_step) == 0:
                ai.calc_mist_lin()
                ai.calc_mist_sq()
                ai.calc_mists_entr()

        # Plotting (mist_corteges (average value of mistakes for array))
        if i % (iterat_num/iterat_step) == 0:
            print(f'{100*i/iterat_num :.1f} %')     # Traning completion percentages
            ai.calc_mist_corteg(len(train_array))


    ai.show_graphics()

    ai.show_data()
    ai.save_data()
    # '''
    # ai.show_graphics()
    # ai.show_data()
