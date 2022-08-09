from constants import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from snake import Snake
from apple import Apple


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


def leaky_relu(x):
    out = np.copy(x)
    out[out > 1] = 1. + 0.01*(out[out >= 1])
    out[out < 0] *= 0.05
    return x


def leaky_relu_deriv(x):
    out = np.copy(x)
    out[(out >= 0) & (out <= 1)] = 1.
    out[out > 1] = 0.01
    out[out < 0] = 0.01
    return out


activ_func = sigmoid
activ_func_deriv = sigmoid_deriv


class AI:
    def __init__(self):
        # Training speed (CONST)
        RANDOM_ARRAYS = 1
        # Training coefficient (CONST)
        self.ALPHA = 0.1
        # Mistakes (graphics)
        self.mist_count = 0
        self.answer_count = 0
        self.mists_prop = np.array([])
        self.mists = np.array([])
        self.mists_corteg = np.array([])
        # Neurons count (CONST)
        self.C_IN = 5   # 3 -- danger detector
                        # 2 -- apple direction
        self.C_H1 = 12
        self.C_H2 = 10
        self.C_OUT = 3  # 4 next direction

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

        if __name__ == "__main__":
            if RANDOM_ARRAYS == 1:
                # Random matrixes
                self.W_in_h1 = np.random.uniform(-1, 1, (self.C_IN, self.C_H1))
                self.W_h1_h2 = np.random.uniform(-1, 1, (self.C_H1, self.C_H2))
                self.W_h2_out = np.random.uniform(-1, 1, (self.C_H2, self.C_OUT))

                self.s_in_h1 = np.random.uniform(-1, 1, (1, self.C_H1))
                self.s_h1_h2 = np.random.uniform(-1, 1, (1, self.C_H2))
                self.s_h2_out = np.random.uniform(-1, 1, (1, self.C_OUT))
            else:
                self.load_data()
        else:
            self.load_data()
        # Mistakes
        self.m_h1 = np.zeros((1, self.C_H1))
        self.m_h2 = np.zeros((1, self.C_H2))
        self.m_out = np.zeros((1, self.C_OUT))


    def run(self, _in):
        self.l_in = np.array(_in).reshape(1,len(_in))
        # calculate output
        self.forward()
        # Choosing commad
        self.choice = np.argmax(self.l_out)


    def training(self, _in, _out):
        # setting in and out arrays
        self.l_in = np.array(_in).reshape(1,len(_in))
        self.l_out_right = np.array(_out).reshape(1,len(_out))

        self.forward()
        self.choice = np.argmax(self.l_out)
        self.backward()
        self.update()


    def forward(self):
        # Finding output
        self.l_h1_row = self.l_in @ self.W_in_h1 + self.s_in_h1
        self.l_h1 = activ_func(self.l_h1_row)
        self.l_h2_row = self.l_h1_row @ self.W_h1_h2 + self.s_h1_h2
        self.l_h2 = activ_func(self.l_h2_row)
        self.l_out_row = self.l_h2_row @ self.W_h2_out + self.s_h2_out
        self.l_out = activ_func(self.l_out_row)


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


    def mutate(self):
        # Random mutate
        # Delta
        self.W_in_h1_mutate = np.random.uniform(-0.1, 0.1, self.W_in_h1.shape)
        self.s_in_h1_mutate = np.random.uniform(-0.1, 0.1, self.s_in_h1)
        self.W_h1_h2_mutate = np.random.uniform(-0.1, 0.1, self.W_h1_h2)
        self.s_h1_h2_mutate = np.random.uniform(-0.1, 0.1, self.s_h1_h2)
        self.W_h2_out_mutate = np.random.uniform(-0.1, 0.1, self.W_h2_out)
        self.s_h2_out_mutate = np.random.uniform(-0.1, 0.1, self.s_h2_out)
        # New weight
        self.W_in_h1 += self.W_in_h1_mutate
        self.s_in_h1 += self.s_in_h1_mutate
        self.W_h1_h2 += self.W_h1_h2_mutate
        self.s_h1_h2 += self.s_h1_h2_mutate
        self.W_h2_out += self.W_h2_out_mutate
        self.s_h2_out += self.s_h2_out_mutate


    def mutete_select(self, prev_score, cur_score):  # score = collected_apple_count ^2 / time
        if (cur_score > prev_score):
            # Save mutation
            return
        # Undo changes
        self.W_in_h1 -= self.W_in_h1_mutate
        self.s_in_h1 -= self.s_in_h1_mutate
        self.W_h1_h2 -= self.W_h1_h2_mutate
        self.s_h1_h2 -= self.s_h1_h2_mutate
        self.W_h2_out -= self.W_h2_out_mutate
        self.s_h2_out -= self.s_h2_out_mutate


    def calc_mist(self):
        mist = 0
        for m in self.m_out[0]:
            mist += m ** 2
        self.mists = np.append(self.mists, mist)


    def calc_mist_corteg(self, last_count):
        mist = 0
        for i in range(last_count):
            mist += self.mists[-i-1]
        mist /= last_count
        self.mists_corteg = np.append(self.mists_corteg, mist)


    def calc_mist_propotion(self):
        self.answer_count += 1
        if self.l_out_right[0,self.choice] == 0:
            self.mist_count += 1
        mist_prop = self.mist_count / self.answer_count
        self.mists_prop = np.append(self.mists_prop, mist_prop)


    def show_graphics(self):
            # Graphics
            figure, axes = plt.subplots(2)
            plt.title('Mistakes')
            # First graphic (count of mistake)
            axes[0].plot(self.mists, 'r.', markeredgewidth = 0)
            axes[0].plot([len(train_array)+i*len(train_array) for i in range(len(self.mists_corteg))], self.mists_corteg, 'b.', markeredgewidth = 0)

            axes[0].text(0.02, 0.94, f"start={'%.3f' % self.mists_corteg[0]}",
                color = 'white', transform=axes[0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0].text(0.25, 0.94, f"end={'%.3f' % self.mists_corteg[-1]}",
                color = 'white', transform=axes[0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0].text(0.02, 0.78, f"min={'%.3f' % self.mists_corteg.min()}",
                color = 'white', transform=axes[0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0].text(0.25, 0.78, f"max={'%.3f' % self.mists_corteg.max()}",
                color = 'white', transform=axes[0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})

            # Second graphics (mist-answer property)
            axes[1].plot(self.mists_prop, 'g.', markeredgewidth = 0)

            axes[1].text(0.02, 0.94, f"answers={self.answer_count}",
                color = 'white', transform=axes[1].transAxes,
                bbox={'facecolor': 'green', 'alpha': 0.85, 'pad': 4})
            axes[1].text(0.35, 0.94, f"mistakes={self.mist_count}",
                color = 'white', transform=axes[1].transAxes,
                bbox={'facecolor': 'green', 'alpha': 0.85, 'pad': 4})
            # Showing
            plt.show()


    def show_data(self):
        print("\nW_in_h1\n", self.W_in_h1)
        print("\nW_h1_h2\n", self.W_h1_h2)
        print("\nW_h2_out\n", self.W_h2_out)
        print("\ns_in_h1\n", self.s_in_h1)
        print("\ns_h1_h2\n", self.s_h1_h2)
        print("\ns_h2_out\n", self.s_h2_out)


    def save_data(self):
        # Save matrixes into the file
        np.savez("Arrays", self.W_in_h1, self.W_h1_h2, self.W_h2_out,
                           self.s_in_h1, self.s_h1_h2, self.s_h2_out)

    def load_data(self):
        # load matrixes from the file
        file = np.load('Arrays.npz')
        self.W_in_h1 = file['arr_0']
        self.W_h1_h2 = file['arr_1']
        self.W_h2_out = file['arr_2']
        self.s_in_h1 = file['arr_3']
        self.s_h1_h2 = file['arr_4']
        self.s_h2_out = file['arr_5']


    def get_choice(self):
        return self.choice-1


    def draw(self):
        pass


if __name__ == "__main__":
    ai = AI()
                    # barrier|apple | choice            (l -- left, f -- forward, r -- right)
                    #  l f r bf lr    l f r
    train_array = [ ( (0,0,0, 0, 0), (1,1,1) ), # Good

                    ( (0,0,0, 1, 0), (0,1,0) ), # Good
                    ( (0,0,0, 1, 1), (0,1,1) ), # Good
                    ( (0,0,0, 0, 1), (0,0,1) ), # Good
                    ( (0,0,0,-1, 1), (0,0,1) ), # Good
                    ( (0,0,0,-1, 0), (1,0,1) ), # Good
                    ( (0,0,0,-1,-1), (1,0,0) ), # Good
                    ( (0,0,0, 0,-1), (1,0,0) ), # Good
                    ( (0,0,0, 1,-1), (1,1,0) ), # Good
                    #  l f r bf lr    l f r
                    ( (0,1,0, 0, 0), (1,0,1) ), # Good

                    ( (0,1,0, 1, 0), (1,0,1) ), # Good
                    ( (0,1,0, 1, 1), (0,0,1) ), # Good
                    ( (0,1,0, 0, 1), (0,0,1) ), # Good
                    ( (0,1,0,-1, 1), (0,0,1) ), # Good
                    ( (0,1,0,-1, 0), (1,0,1) ), # Good
                    ( (0,1,0,-1,-1), (1,0,0) ), # Good
                    ( (0,1,0, 0,-1), (1,0,0) ), # Good
                    ( (0,1,0, 1,-1), (1,0,0) ), # Good
                    #  l f r bf lr    l f r
                    ( (1,0,0, 0, 0), (0,1,1) ), # Good

                    ( (1,0,0, 1, 0), (0,1,0) ), # Good
                    ( (1,0,0, 1, 1), (0,1,1) ), # Good
                    ( (1,0,0, 0, 1), (0,0,1) ), # Good
                    ( (1,0,0,-1, 1), (0,0,1) ), # Good
                    ( (1,0,0,-1, 0), (0,0,1) ), # Good
                    ( (1,0,0,-1,-1), (0,1,0) ), # Good
                    ( (1,0,0, 0,-1), (0,1,0) ), # Good
                    ( (1,0,0, 1,-1), (0,1,0) ), # Good
                    #  l f r bf lr    l f r
                    ( (0,0,1, 0, 0), (1,1,0) ), # Good

                    ( (0,0,1, 1, 0), (0,1,0) ), # Good
                    ( (0,0,1, 1, 1), (0,1,0) ), # Good
                    ( (0,0,1, 0, 1), (0,1,0) ), # Good
                    ( (0,0,1,-1, 1), (0,1,0) ), # Good
                    ( (0,0,1,-1, 0), (1,0,0) ), # Good
                    ( (0,0,1,-1,-1), (1,0,0) ), # Good
                    ( (0,0,1, 0,-1), (1,0,0) ), # Good
                    ( (0,0,1, 1,-1), (1,1,0) ), # Good
                    #  l f r bf lr    l f r
                    ( (1,1,0, 0, 0), (0,0,1) ), # Good

                    ( (1,1,0, 1, 0), (0,0,1) ), # Good
                    ( (1,1,0, 1, 1), (0,0,1) ), # Good
                    ( (1,1,0, 0, 1), (0,0,1) ), # Good
                    ( (1,1,0,-1, 1), (0,0,1) ), # Good
                    ( (1,1,0,-1, 0), (0,0,1) ), # Good
                    ( (1,1,0,-1,-1), (0,0,1) ), # Good
                    ( (1,1,0, 0,-1), (0,0,1) ), # Good
                    ( (1,1,0, 1,-1), (0,0,1) ), # Good
                    #  l f r bf lr    l f r
                    ( (0,1,1, 0, 0), (1,0,0) ), # Good

                    ( (0,1,1, 1, 0), (1,0,0) ), # Good
                    ( (0,1,1, 1, 1), (1,0,0) ), # Good
                    ( (0,1,1, 0, 1), (1,0,0) ), # Good
                    ( (0,1,1,-1, 1), (1,0,0) ), # Good
                    ( (0,1,1,-1, 0), (1,0,0) ), # Good
                    ( (0,1,1,-1,-1), (1,0,0) ), # Good
                    ( (0,1,1, 0,-1), (1,0,0) ), # Good
                    ( (0,1,1, 1,-1), (1,0,0) ), # Good
                    #  l f r bf lr    l f r
                    ( (1,0,1, 0, 0), (0,1,0) ), # Good

                    ( (1,0,1, 1, 0), (0,1,0) ), # Good
                    ( (1,0,1, 1, 1), (0,1,0) ), # Good
                    ( (1,0,1, 0, 1), (0,1,0) ), # Good
                    ( (1,0,1,-1, 1), (0,1,0) ), # Good
                    ( (1,0,1,-1, 0), (0,1,0) ), # Good
                    ( (1,0,1,-1,-1), (0,1,0) ), # Good
                    ( (1,0,1, 0,-1), (0,1,0) ), # Good
                    ( (1,0,1, 1,-1), (0,1,0) )  # Good
                    ]

    # Testing
    '''
    for set in train_array:
        ai.run(set[0])
        print(f"input: {set[0]}\noutput: {set[1]}\nchoice: {ai.get_choice()}\n")
    '''

    # Training
    # '''
    num_iterat = 10000
    # 10k -- ~ 60 sec ~ -- works
    # 100k -- ~ 10 min ~
    # 1 million -- ~ 1-2 hours ~ -- perfectly, but train long time
    for i in range(num_iterat):
        random.shuffle(train_array)
        for set in train_array:
            ai.training(set[0], set[1])
            # For graphics (mists, propotion)
            if i % (num_iterat//1000) == 0:
                ai.calc_mist()
                ai.calc_mist_propotion()
        # For graphics (mist_corteges)
        if i % (num_iterat//1000) == 0:
            print(f'{100*i/num_iterat :.1f} %')     # Traning comletion percentages
            ai.calc_mist_corteg(len(train_array))

    ai.show_graphics()
    # '''

    # ai.show_data()
    # ai.save_data()
