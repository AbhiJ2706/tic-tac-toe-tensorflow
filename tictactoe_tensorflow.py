'''
Abhi Jain

October 10th 2019

Neural network plays tic-tac-toe

This program uses a neural network to play tictactoe. It uses cases in which a player is about to win, custom created by me using a database of games that were already won
by a player, modified to open a winning move for the computer to choose. The board is numerically stored as follows:
0- empty space
1- space occupied by player
-1- space occupied by computer
spaces:
[0,1,2,
 3,4,5,
 6,7,8]
For the first 4 weeks of this project I spent time with a different file, which used straight math (so each equation was written out in code) rather than Tensorflow.
This quickly became super hard to manage and really hard to modify (since I don't know calculus) so I switched over to Tensorflow, which let you choose the formulas
and handles the math on its own. This proved to be much easier to use and modify, and I spent the last 2 weeks tweaking the parameters and the databases for optimal play.
This final copy of the project obtains this, as it recognizes where the opponent is about to make a winning move, and can recognize when it is about to win. Since
the computer never goes first, it cannot win in most cases. The computer is trained to rarely, if ever, lose.

'''

#import libraries
#machine learning libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
#other libraries
import numpy as np
import random
import collections

print(tf.__version__)

#loading database
A = np.loadtxt("data.txt")
#separating questions from answers using columns
X = A[:,:9]      
y = A[:,9:]

#checks in each example which player (X or O) wins
def checkforWin(x, board):
    if board[0] == x and board[1] == x and board[2] == x: #win case
        tempBoard = [0, 1, 2] #array containing win case
        return 1, tempBoard #returns result and win case
    if board[3] == x and board[4] == x and board[5] == x:
        tempBoard = [3, 4, 5]
        return 2, tempBoard
    if board[6] == x and board[7] == x and board[8] == x:
        tempBoard = [6, 7, 8]
        return 3, tempBoard
    if board[0] == x and board[3] == x and board[6] == x:
        tempBoard = [0, 3, 6]
        return 4, tempBoard
    if board[1] == x and board[4] == x and board[7] == x:
        tempBoard = [1, 4, 7]
        return 5, tempBoard
    if board[2] == x and board[5] == x and board[8] == x:
        tempBoard = [2, 5, 8]
        return 6, tempBoard
    if board[0] == x and board[4] == x and board[8] == x:
        tempBoard = [0, 4, 8]
        return 7, tempBoard
    if board[2] == x and board[4] == x and board[6] == x:
        tempBoard = [2, 4, 6]
        return 8, tempBoard
    else:
        return 0

#going through cases and removing a winning move to create cases
for i in range(len(X)):
    #checking for error
    try:
        result, tempBoard = checkforWin(y[i], X[i]) #figuring out who wins
        if result > 0:
            j = random.randint(0,2)
            y[i] = tempBoard[j] #setting the answers to be the removed winning move
            X[i][int(y[i])] = 0 #removing a winning move
            a = collections.Counter(X[i]) #counting amount of each piece 
            x = a[1] #counting X's
            Y = a[-1] #counting O's
            if x == Y:
                #removing a random O so that it is O's turn
                indices = [k for k, x in enumerate(X[i]) if x == -1]
                l = random.randint(0,len(indices)-1)
                X[i][indices[l]] = 0          
    except TypeError:
        pass

#some cases return -1 as the answer. This removes them and replaces them with a viable case
for i in range(len(X)):
    if y[i] == -1:
        X[i] = [0,0,0,0,0,0,0,0,0]
        y[i] = 0

(train_images, train_labels), (test_images, test_labels) = (X, y), (X, y) #making training & test cases

class_names = [0,1,2,3,4,5,6,7,8] #list of different possible answers

#neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(9,)), #input layer
    keras.layers.Dense(18, activation=tf.nn.tanh), #hidden layer
    keras.layers.Dense(9, activation=tf.nn.softmax) #output layer
])

#compiling neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

#training model
model.fit(train_images, train_labels, epochs=2000)

#running tests (no learning occurs here)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#using network for predictions
predictions = model.predict(test_images)

#printing predictions
for i in range(15):
    print(np.round(predictions[i]*1000)/1000, np.argmax(predictions[i]), test_labels[i])

#Tic tac toe actual game
import pygame, sys
from pygame.locals import *

#initialise pygame and screen
pygame.init()
surface = pygame.display.set_mode((480,480))
color = (0,0,0)
surface.fill((255,255,255))
pygame.draw.line(surface, color, (160,0), (160,480), 4)
pygame.draw.line(surface, color, (320,0), (320,480), 4)
pygame.draw.line(surface, color, (0,160), (480,160), 4)
pygame.draw.line(surface, color, (0,320), (480,320), 4)

#mapping screen
zones = []
for i in range(3):
    for j in range(3):
        zones.append([j*160,i*160,j*160+160,i*160+160])

#board positions
board = [0,0,0,0,0,0,0,0,0]

#checking for a win in the actual tic tac toe game
def checkForWin(x):
    if board[0] == x and board[1] == x and board[2] == x:
        return True
    if board[3] == x and board[4] == x and board[5] == x:
        return True
    if board[6] == x and board[7] == x and board[8] == x:
        return True
    if board[0] == x and board[3] == x and board[6] == x:
        return True
    if board[1] == x and board[4] == x and board[7] == x:
        return True
    if board[2] == x and board[5] == x and board[8] == x:
        return True
    if board[0] == x and board[4] == x and board[8] == x:
        return True
    if board[2] == x and board[4] == x and board[6] == x:
        return True
    else:
        return False

#tictac toe game loop
def mainloop():
    mainloop = True
    while mainloop== True:
        pygame.draw.line(surface, color, (160,0), (160,480), 4)
        pygame.draw.line(surface, color, (320,0), (320,480), 4)
        pygame.draw.line(surface, color, (0,160), (480,160), 4)
        pygame.draw.line(surface, color, (0,320), (480,320), 4)
        #getting click events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                mainloop = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mousex, mousey = pygame.mouse.get_pos()
                for i in range(9):
                    #playing game
                    if mousex >= zones[i][0] and mousex <= zones[i][2] and mousey >= zones[i][1] and mousey <= zones[i][3]: #figures out what zone the mouse clicked in
                        if board[i] == 0:
                            board[i] = 1 #filling board position
                            #drawing X
                            mid = (int((zones[i][0] + zones[i][2])/2), int((zones[i][1] + zones[i][3])/2))
                            pygame.draw.line(surface, color, (mid[0]-60, mid[1]-60), (mid[0]+60, mid[1]+60), 4)
                            pygame.draw.line(surface, color, (mid[0]+60, mid[1]-60), (mid[0]-60, mid[1]+60), 4)
                            print(np.array([board], dtype="float"))
                            #checks if either player won or draw
                            xwin = checkForWin(1)
                            ywin = checkForWin(-1)
                            if xwin:
                                print("X wins!")
                                mainloop = False
                                break
                            elif 0 not in board and not ywin and not xwin:
                                print("Draw!")
                                mainloop = False
                                break
                            #neural network runs a prediction to get the best move
                            predictions = model.predict(np.array([board], dtype="float"))
                            print(predictions)
                            print(np.argmax(predictions))
                            space = np.argmax(predictions)
                            board[space] = -1 #populating board
                            #drawing
                            mid = (int((zones[space][0] + zones[space][2])/2), int((zones[space][1] + zones[space][3])/2))
                            pygame.draw.circle(surface, color, mid, 60, 4)
                            break
                #checking for win for O
                ywin = checkForWin(-1)
                if ywin:
                    print("Y wins!")
                    mainloop = False
                    break
        pygame.display.flip()
        if not mainloop:
            break

#putting everything together
while 1:
    mainloop();
    if input("Play again?: ") == "no":
        break;
    board = [0,0,0,0,0,0,0,0,0]
    surface.fill((255,255,255))
    pygame.draw.line(surface, color, (160,0), (160,480), 4)
    pygame.draw.line(surface, color, (320,0), (320,480), 4)
    pygame.draw.line(surface, color, (0,160), (480,160), 4)
    pygame.draw.line(surface, color, (0,320), (480,320), 4)
    
pygame.quit()



