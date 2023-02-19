import time
import sys
import gzip
import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from RandomNavigator import RandomNavigator
from greedy_navigator import GreedyNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

# Create a Map Class Object
map = Map()
map.getNewMap()
map.getNewMap()
map.getNewMap()
map.getNewMap()
map.getNewMap()
map.getNewMap()
# map.getNewMap()
# map.getNewMap()
# map.getNewMap()
# Get the current map from the Map Class
data = map.map
# Print the number of the current map
print(map.number)
# Create a Robot that starts at (0,0)
# The Robot Class stores the current position of the robot
# and provides ways to move the robot
robot = Robot(0, 0)
navigator = GreedyNavigator()
# Create a Game object, providing it with the map data, the goal location of the map, the navigator, and the robot
game = Game(data, map.number, navigator, robot)

# This loop runs the game for 1000 ticks, stopping if a goal is found.
np.set_printoptions(suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize)
for i in range(300):
    found_goal = game.tick()
    print(f"{game.getIteration()}: Robot at: {robot.getLoc()}, Score = {game.getScore()}")
    if found_goal:
        print(f"Found goal at time step: {game.getIteration()}!")
        break
print(f"Final Score: {game.score}")

# Image.fromarray(image).show()
im = Image.fromarray(np.uint8(game.exploredMap)).show()
map.getNewMap()

# print(game.exploredMap)
# x = robot.getLoc()
# print(x)
# # RIGHT
# #print(game.exploredMap[:, x[0]:])
# # LEFT
# #print(game.exploredMap[:, :x[0]])
# # DOWN
# #print(game.exploredMap[x[1]:, :])
# # UP
# #print(game.exploredMap[:x[1], :])
# right_info = np.sum(game.exploredMap[:, x[0]:])
# left_info = np.sum(game.exploredMap[:, :x[0]])
# down_info = np.sum(game.exploredMap[x[1]:, :])
# up_info = np.sum(game.exploredMap[:x[1], :])

# print("RIGHT", right_info)
# print("LEFT", left_info)
# print("DOWN", down_info)
# print("UP", up_info)


# print(game.exploredMap[2][0])
# print(game.exploredMap[0][2])
# # # Show how much of the world has been explored
# # # Create the world estimating network
# uNet = WorldEstimatingNetwork()
#  # Create the digit classification network
#  classNet = DigitClassificationNetwork()
#   # This loop shows how you can create a mask, an grid of 0s and 1s
#   # where 0s represent unexplored areas and 1s represent explored areas
#   # This mask is used by the world estimating network
#   mask = np.zeros((28, 28))
#    for col in range(0, 28):
#         for row in range(0, 28):
#             if game.exploredMap[col, row] != 128:
#                 mask[col, row] = 1

#     # Creates an estimate of what the world looks like
#     image = uNet.runNetwork(game.exploredMap, mask)

# # # Show the image of the estimated world
# # Image.fromarray(image).show()
# # # Use the classification network on the estimated image
# # # to get a guess of what "world" we are in (e.g., what the MNIST digit of the world)
#     char = classNet.runNetwork(image)
#     # # get the most likely digit

#     def softmax(x):
#         return(np.exp(x)/np.exp(x).sum())

#     print(char.argmax())
#     print(softmax(char))
#     print(np.sum(softmax(char)))
# # Get the next map to test your algorithm on.
# print(image)
# Image.fromarray(image).show()
# im = Image.fromarray(np.uint8(game.exploredMap)).show()
# map.getNewMap()
