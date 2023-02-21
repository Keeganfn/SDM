import time
import sys
import gzip
import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from RandomNavigator import RandomNavigator
from greedy_navigator import GreedyNavigator
from branch_bound_navigator import BBNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

# Create a Map Class Object
map = Map()
actual_number = [6, 1, 3, 9, 5, 2, 4, 0, 7, 8]
image_number = [201, 202, 205, 206, 207, 208, 210, 215, 220, 226]

# This loop runs the game for 1000 ticks, stopping if a goal is found.
np.set_printoptions(suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize)
avg = []
map_num = []
runtime = []
for k in range(10):
    robot = Robot(0, 0)
    map.imageNumber = image_number[k] - 1
    map.getNewMap()
    #navigator = BBNavigator()
    navigator = GreedyNavigator()
    data = map.map
    map_num.append(int(map.number))
    game = Game(data, map.number, navigator, robot)
    print("Map Number: ", map.number)
    start = time.time()
    for i in range(500):
        found_goal = game.tick()
        #print(f"{game.getIteration()}: Robot at: {robot.getLoc()}, Score = {game.getScore()}")
        if found_goal:
            print(f"Found goal at time step: {game.getIteration()}!")
            break
    end = time.time()
    runtime.append(end-start)
    print("Total Runtime: ", end-start)
    im = Image.fromarray(np.uint8(game.exploredMap)).show()
    print(f"Final Score: {game.score}")
    avg.append(game.score)

print(avg)
print(map_num)
print(runtime)
print("AVERAGE SCORE = ", sum(avg) / len(avg))
print("AVERAGE TIME = ", sum(runtime) / len(runtime))
