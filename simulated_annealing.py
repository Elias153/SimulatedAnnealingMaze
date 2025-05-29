import random
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from bokeh.core.property.nullable import Nullable


class SimulatedAnnealing:

    def __init__(self):
        # Labyrinth: 0 = frei, 1 = Wand

        self.maze = [
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]

        self.start = (0, 0)
        self.goal = (9, 9)

        self.maze = [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ]
        self.start = (1, 1)
        self.goal = (18, 18)

        self.rows = len(self.maze)
        self.cols = len(self.maze[0])
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def euclidean(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_neighbors(self, pos):
        neighbors = []
        for d in self.directions:
            new_r, new_c = pos[0] + d[0], pos[1] + d[1]
            if 0 <= new_r < self.rows and 0 <= new_c < self.cols and self.maze[new_r][new_c] == 0:
                neighbors.append((new_r, new_c))
        return neighbors

    def run(self, max_iter=100000, temp=1000, alpha=0.99):
        current = self.start
        parent = {self.start: None}

        for _ in range(max_iter):
            if current == self.goal:
                break

            neighbors = self.get_neighbors(current)
            if not neighbors:
                return None

            next_pos = random.choice(neighbors)
            delta = self.euclidean(current, self.goal) - self.euclidean(next_pos, self.goal)

            if delta > 0 or random.random() < math.exp(-abs(delta) / temp):
                # Only update parent if move is accepted
                if next_pos not in parent:
                    parent[next_pos] = current
                current = next_pos

            temp *= alpha
            if temp < 1e-3:
                break

        # Reconstruct clean path
        if current == self.goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        else:
            return None

    def evaluate_sa(self,algorithm, runs=100):
        plot_first_success = True
        successes = 0
        path_lengths = []
        times = []

        for _ in range(runs):
            start_time = time.time()
            path = algorithm.run()
            duration = time.time() - start_time

            if path:
                if plot_first_success:
                    self.plot_maze_path(path)
                    plot_first_success = False
                successes += 1
                path_lengths.append(len(path))
                times.append(duration)

        success_rate = successes / runs
        avg_path_length = np.mean(path_lengths) if path_lengths else None
        avg_time = np.mean(times) if times else None

        return {
            "success_rate": success_rate,
            "average_path_length": avg_path_length,
            "average_time": avg_time,
            "successful_runs": successes,
        }

    def plot_maze_path(self, path):
        plt.figure(figsize=(6, 5))
        for r in range(self.rows):
            for c in range(self.cols):
                color = 'black' if self.maze[r][c] == 1 else 'white'
                plt.gca().add_patch(plt.Rectangle((c, self.rows - r - 1), 1, 1, facecolor=color, edgecolor='gray'))

        if path:
            for (r, c) in path:
                plt.gca().add_patch(plt.Rectangle((c, self.rows - r - 1), 1, 1, facecolor='lightblue'))

            sr, sc = path[0]
            gr, gc = path[-1]
            plt.gca().add_patch(plt.Rectangle((sc, self.rows - sr - 1), 1, 1, facecolor='green'))  # Start
            plt.gca().add_patch(plt.Rectangle((gc, self.rows - gr - 1), 1, 1, facecolor='red'))    # Goal

        plt.xlim(0, self.cols)
        plt.ylim(0, self.rows)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Instanziierung und Ausführung
    sa = SimulatedAnnealing()
    results = sa.evaluate_sa(sa, runs=100)

    success_rate = results["success_rate"]
   # average_path_length = results["average_path_length"]

    if success_rate is not None and success_rate != 0:
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Average Path Length: {results['average_path_length']}")
        print(f"Average Time per Run: {results['average_time']:.5f} seconds")
        print(f"Successful Runs: {results['successful_runs']}")
    else:
        print("No success rate could be found, as no run was successful.")



    #path = sa.run()
    #sa.plot_maze_path(path)

