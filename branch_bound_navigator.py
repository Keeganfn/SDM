import numpy as np
from random import randint
import time
import sys
from PIL import Image
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork
from copy import deepcopy


class BBNavigator:
    def __init__(self):
        # Create the world estimating network
        self.uNet = WorldEstimatingNetwork()
        # Create the digit classification network
        self.classNet = DigitClassificationNetwork()
        self.robot_past_locations = []
        self.goal = 0
        self.finish_flag = False
        self.iteration = 0
        self.iteration2 = 0
        self.current_best_info_score = -100000
        self.current_best_path = None
        self.guesses = []
        self.num_it = 5
        self.area = 10
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

    def get_surroundings(self, start, robot):
        locations = []
        if robot.checkValidLoc(x=start[0]+1, y=start[1]):
            right_loc = (start[0]+1, start[1])
            locations.append(right_loc)
        if robot.checkValidLoc(x=start[0]-1, y=start[1]):
            left_loc = (start[0]-1, start[1])
            locations.append(left_loc)
        if robot.checkValidLoc(x=start[0], y=start[1]+1):
            down_loc = (start[0], start[1]+1)
            locations.append(down_loc)
        if robot.checkValidLoc(x=start[0], y=start[1]-1):
            up_loc = (start[0], start[1]-1)
            locations.append(up_loc)
        return locations

    def get_manhattan_distance(self, start, end):
        return abs(end[0] - start[0]) + abs(end[1] - start[1])

    def bb_ipp(self, start, end, B, path_so_far, robot, info_score, image):
        new_path = None
        is_leaf = True
        surrounding_nodes = self.get_surroundings(start, robot)
        for i in range(len(surrounding_nodes)):
            if surrounding_nodes[i] == start:
                surrounding_nodes.pop(i)

        for node in surrounding_nodes:
            if node not in path_so_far.keys():
                new_path = deepcopy(path_so_far)
                new_path[start] = node
                B_new = B - 1
                if self.get_manhattan_distance(node, end) <= B_new:
                    is_leaf = False
                    new_info_score = info_score
                    information_surroundings = self.get_surroundings(node, robot)
                    for i in range(len(information_surroundings)):
                        if information_surroundings[i] == start:
                            information_surroundings.pop(i)
                            break
                    for i in range(len(information_surroundings)):
                        new_info_score += image[information_surroundings[i][1], information_surroundings[i][0]]
                    if new_info_score > self.current_best_info_score:
                        self.bb_ipp(node, end, B_new, new_path, robot, new_info_score, image)
        if is_leaf:
            if info_score > self.current_best_info_score and start == end:
                self.current_best_info_score = info_score
                self.current_best_path = deepcopy(path_so_far)

    def get_recreated_path(self, dictionary):
        pass

    def getAction(self, robot, map):
        self.iteration += 1
        self.iteration2 += 1
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
                if 1 not in self.guesses:
                    self.goal = 3
                elif 2 not in self.guesses:
                    self.goal = 7
                else:
                    self.goal = 0
                if self.goal >= 0 and self.goal <= 2:
                    self.guesses.append(0)
                    dir = self.go_top_right(location)
                if self.goal >= 2 and self.goal <= 5:
                    self.guesses.append(1)
                    dir = self.go_bottom_right(location)
                if self.goal >= 6:
                    self.guesses.append(2)
                    dir = self.go_bottom_left(location)
            return dir

        # get the most likely digit
        for i in range(len(char[0])):
            if char[0][i] > .8 and self.iteration2 > 51:
                print(char[0][i])
                print(i)
                self.goal = i
                self.finish_flag = True

        for i in self.robot_past_locations:
            image[i[1]][i[0]] -= 10000

        if self.iteration % self.num_it == 0 or self.iteration == 1:
            self.current_best_info_score = -10000000
            self.current_best_path = None
            mask = np.zeros((28, 28))
            current_max_dist = 0
            selected = None
            area = 8
            test2 = []
            test = []
            for i in range(max(location[1]-area, 0), min(location[1]+area, 27)):
                for j in range(max(location[0]-area, 0), min(location[0]+area, 27)):
                    test.append((j, i))
                    test2.append(image[i][j])

            test2 = np.array(test2)
            selected = test[test2.argmax()]

            goal = np.unravel_index(np.argmax(image), image.shape)
            goal = (goal[1], goal[0])
            if selected is not None and selected != location:
                goal = selected
            total = self.get_manhattan_distance(location, goal)
            self.num_it = total
            self.bb_ipp(start=location, end=goal, B=total, path_so_far={}, robot=robot, info_score=0, image=image)
            if len(self.current_best_path) < self.num_it:
                self.iteration = self.num_it - len(self.current_best_path)

        while direction is None:
            best_dir = self.current_best_path[location]
            best_dir = (best_dir[0] - location[0], best_dir[1]-location[1])
            if best_dir == (-1, 0):
                direction = 'left'
            if best_dir == (1, 0):
                direction = 'right'
            if best_dir == (0, 1):
                direction = 'down'
            if best_dir == (0, -1):
                direction = 'up'

            # If it is not a valid move, reset
            if not robot.checkValidMove(direction):
                direction = None
        # print(direction)
        return direction
