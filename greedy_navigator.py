import numpy as np
from random import randint
import time
import sys
from PIL import Image
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork


class GreedyNavigator:
    def __init__(self):
        # Create the world estimating network
        self.uNet = WorldEstimatingNetwork()
        # Create the digit classification network
        self.classNet = DigitClassificationNetwork()
        self.robot_past_locations = []
        self.goal = 0
        self.finish_flag = False
        self.iteration = 0
        self.guesses = []
        pass

    # Got softmax implementation from https://www.sharpsightlabs.com/blog/numpy-softmax/
    def softmax(self, x):
        return(np.exp(x)/np.exp(x).sum())

    def go_top_right(self, location):
        if location[0] > 0:
            return "left"
        if location[1] < 27:
            return "down"
        if location[1] == 27 and location[0] == 0:
            return None

    def go_bottom_right(self, location):
        if location[0] < 27:
            return "right"
        if location[1] < 27:
            return "down"
        if location[1] == 27 and location[0] == 27:
            return None

    def go_bottom_left(self, location):
        if location[1] > 0:
            return "up"
        if location[0] < 27:
            return "right"
        if location[1] == 0 and location[0] == 27:
            return None

    def getAction(self, robot, map):
        self.iteration += 1
        direction = None
        location = robot.getLoc()
        self.robot_past_locations.append(location)
        # This mask is used by the world estimating network
        mask = np.zeros((28, 28))
        for col in range(0, 28):
            for row in range(0, 28):
                if map[col, row] != 128:
                    mask[col, row] = 1

       # Creates an estimate of what the world looks like
        image = self.uNet.runNetwork(map, mask)
        # print(image)
        char = self.classNet.runNetwork(image)

        char = self.softmax(char)

        if self.finish_flag:
            if self.goal >= 0 and self.goal <= 2:
                self.guesses.append(0)
                dir = self.go_top_right(location)
            if self.goal > 2 and self.goal <= 5:
                self.guesses.append(1)
                dir = self.go_bottom_right(location)
            if self.goal >= 6:
                self.guesses.append(2)
                dir = self.go_bottom_left(location)

            if dir == None:
                print(char[0].argmax())
                print("NOT")
                if 1 not in self.guesses:
                    self.goal = 3
                elif 2 not in self.guesses:
                    self.goal = 7
                else:
                    self.goal = 0
                if self.goal >= 0 and self.goal <= 2:
                    print("HERE1")
                    self.guesses.append(0)
                    dir = self.go_top_right(location)
                if self.goal >= 2 and self.goal <= 5:
                    print("HERE2")
                    self.guesses.append(1)
                    dir = self.go_bottom_right(location)
                if self.goal >= 6:
                    print("HERE3")
                    self.guesses.append(2)
                    dir = self.go_bottom_left(location)
            return dir

        # get the most likely digit
        for i in range(len(char[0])):
            if char[0][i] > .8 and self.iteration > 51:
                print(char[0][i])
                print(i)
                self.goal = i
                self.finish_flag = True

        # print(char.argmax())
        # print(self.softmax(char))

        for i in self.robot_past_locations:
            image[i[1]][i[0]] -= 100

        # finds the direction with the most potential information gain
        left_info, right_info, down_info, up_info = -1000000000, -1000000000, -1000000000, -1000000000
        if robot.checkValidMove("right"):
            right_info = image[location[1]][location[0]+1]
        if robot.checkValidMove("left"):
            left_info = image[location[1]][location[0]-1]
        if robot.checkValidMove("down"):
            down_info = image[location[1]+1][location[0]]
        if robot.checkValidMove("up"):
            up_info = image[location[1]-1][location[0]]

        direction_map = np.array([left_info, right_info, down_info, up_info])

        while direction is None:

            best_dir = np.argmax(direction_map)
            if best_dir == 0:
                direction = 'left'
            if best_dir == 1:
                direction = 'right'
            if best_dir == 2:
                direction = 'down'
            if best_dir == 3:
                direction = 'up'

            # If it is not a valid move, reset
            if not robot.checkValidMove(direction):
                direction_map[best_dir] = -100000000000000000
                direction = None
        # print(direction)
        return direction
