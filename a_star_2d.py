import numpy as np
import math
import time
import priority_queue
from maze import Maze2D


class AStar2D():
    def __init__(self, map_name="maze2.pgm", start=(1, 1), end=(24, 24)) -> None:
        #self.maze = Maze2D(start)
        self.maze = Maze2D.from_pgm(map_name)
        self.start = self.maze.get_start()
        self.end = self.maze.get_goal()

    def euclidean_heuristic(self, start, end) -> float:
        start = self.maze.state_from_index(start)
        end = self.maze.state_from_index(end)
        return math.sqrt(((end[0]-start[0])**2) + ((end[1]-start[1])**2))

    def get_path(self, come_from, current):
        complete_path = []
        while current in come_from.keys():
            complete_path.append(self.maze.state_from_index(current))
            current = come_from[current]
        complete_path.reverse()
        return complete_path

    # Using algorithm outline found on Wikipedia: https://en.wikipedia.org/wiki/A*_search_algorithm
    def find_path(self):
        prio_q = priority_queue.PriorityQueue()
        prio_q.insert(self.start, 0)
        come_from = {self.start: None}
        path_cost = {self.start: 0}

        while prio_q.size != 0:
            current = prio_q.pop()
            if current == self.end:
                print("Complete")
                final = self.get_path(come_from, current)
                self.maze.plot_path(final, "Maze2D")
                return final

            for next in self.maze.get_neighbors(current):
                cost = path_cost[current] + self.euclidean_heuristic(current, next)
                if next not in path_cost or cost < path_cost[next]:
                    path_cost[next] = cost
                    prio_q.insert(next, cost + self.euclidean_heuristic(self.end, next))
                    come_from[next] = current

    def find_path_benchmark(self, epsilon=10, max_time=.05):
        while epsilon != 1:
            if epsilon < 1.001:
                epsilon = 1
            prio_q = priority_queue.PriorityQueue()
            prio_q.insert(self.start, 0)
            come_from = {self.start: None}
            path_cost = {self.start: 0}
            start = time.time() + max_time
            nodes_expanded = 0
            final = 0
            while prio_q.size != 0:
                current = prio_q.pop()
                if current == self.end:
                    print("Complete")
                    final = self.get_path(come_from, current)
                    #self.maze.plot_path(final, "Maze2D")
                    print(f"EPSILON: {epsilon}\nNODES EXPANDED: {nodes_expanded}\nPATH LENGTH: {len(final)}")

                if time.time() > start:
                    print("DID NOT COMPLETE")
                    break

                for next in self.maze.get_neighbors(current):
                    nodes_expanded += 1
                    cost = path_cost[current] + self.euclidean_heuristic(current, next) * epsilon
                    if next not in path_cost or cost < path_cost[next]:
                        path_cost[next] = cost
                        prio_q.insert(next, cost + self.euclidean_heuristic(self.end, next))
                        come_from[next] = current
            epsilon = epsilon - (.5 * (epsilon-1))


if __name__ == "__main__":
    ast = AStar2D()
    #path = ast.find_path()
    path = ast.find_path_benchmark()
    print(path)
