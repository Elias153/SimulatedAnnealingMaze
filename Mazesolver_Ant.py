import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time

# Festes Maze laut Benutzervorgabe
#MAZE_DATA = [    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
   #              [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
   #              [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
   #              [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
   #              [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
   #              [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
   #              [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
   #              [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
   #              [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
   #              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
   #              [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
   #              [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
   #              [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
   #              [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
   #              [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
   #              [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
   #              [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
   #              [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
   #              [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
   #              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
#START = (1, 1)
#GOAL = (18, 18)

MAZE_DATA = [
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

START = (0, 0)
GOAL = (9, 9)


def get_neighbors(pos, maze):
    x, y = pos
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
            if maze[nx][ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

def euclidean(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def ant_colony(maze, start, goal, num_ants=20, num_iterations=50, alpha=5, beta=1, evaporation=0.5, q=100):
    pheromone = np.ones_like(maze, dtype=float)
    best_path = []
    best_length = float('inf')

    for iteration in range(num_iterations):
       # print(f"Iteration {iteration+1}/{num_iterations}")
        all_paths = []
        for ant in range(num_ants):
            path = [start]
            visited = set(path)
            current = start
            while current != goal:
                neighbors = get_neighbors(current, maze)
                neighbors = [n for n in neighbors if n not in visited]
                if not neighbors:
                    break
                probs = []
                debug_choices = []
                for n in neighbors:
                    tau = pheromone[n]
                    eta = 1 / (euclidean(n, goal) + 1e-6)
                    p = (tau ** alpha) * (eta ** beta)
                    probs.append(p)
                    debug_choices.append((n, round(p, 4)))
                probs = np.array(probs)
                probs /= probs.sum()
                next_pos = neighbors[np.random.choice(len(neighbors), p=probs)]
                #if ant < 3:
                 #   print(f"  Ameise {ant+1}: {current} -> {next_pos} | Optionen: {debug_choices}")
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
            if path[-1] == goal and len(path) < best_length:
                best_path = path
                best_length = len(path)
            all_paths.append(path)

        pheromone *= (1 - evaporation)
        for path in all_paths:
            if path[-1] == goal:
                for pos in path:
                    pheromone[pos] += q / len(path)

    return best_path, pheromone



def draw_maze(maze, path, start, goal, pheromone=None, title_suffix=""):
    rows = len(maze)
    cols = len(maze[0])
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    for r in range(rows):
        for c in range(cols):
            color = 'black' if maze[r][c] == 1 else 'white'
            ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=color, edgecolor='gray'))

    if path:
        for (r, c) in path:
            ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor='lightblue'))
        sr, sc = path[0]
        gr, gc = path[-1]
        ax.add_patch(plt.Rectangle((sc, rows - sr - 1), 1, 1, facecolor='green'))  # Start
        ax.add_patch(plt.Rectangle((gc, rows - gr - 1), 1, 1, facecolor='red'))    # Goal

    plt.xlim(0, cols)
    plt.ylim(0, rows)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title(f"Maze Pfad {title_suffix}")
    plt.savefig(f'maze_path_{title_suffix}.png')
    plt.show()

    if pheromone is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(pheromone, cmap='hot')
        plt.title(f'Pheromonkarte {title_suffix}')
        plt.colorbar(label='Pheromonstärke')
        plt.savefig(f'pheromone_map_{title_suffix}.png')
        plt.show()


class MazeACO:
    def __init__(self):
        self.maze = np.array(MAZE_DATA)
        self.start = START
        self.goal = GOAL

    def run(self):
        path, _ = ant_colony(self.maze, self.start, self.goal)
        return path


    def evaluate(self,runs=10):
        first_run= True

        total_path_length = 0
        total_time = 0
        successful_runs = 0

        for i in range(runs):
            print(f"Lauf {i + 1}/{runs}...")
            maze_solver = MazeACO()
            start_time = time.time()
            path = maze_solver.run()
            end_time = time.time()

            if path and path[-1] == GOAL:
                if first_run:
                    self.plot_maze_path(path)
                    first_run = False
                successful_runs += 1
                total_path_length += len(path)
                total_time += (end_time - start_time)

        success_rate = successful_runs / runs
        average_path_length = total_path_length / successful_runs if successful_runs > 0 else None
        average_time = total_time / successful_runs if successful_runs > 0 else None

        return {
            "success_rate": success_rate,
            "average_path_length": average_path_length,
            "average_time": average_time,
            "successful_runs": successful_runs
        }

    def plot_maze_path(self, path):
        draw_maze(self.maze, path, self.start, self.goal, title_suffix="main")

if __name__ == "__main__":
    solver = MazeACO()
    results = solver.evaluate(runs=100)

    if results["success_rate"]:
        print(f"Success Rate: {results['success_rate']:.2f}")
        print(f"Average Path Length: {results['average_path_length']}")
        print(f"Average Time per Run: {results['average_time']:.5f} seconds")
        print(f"Successful Runs: {results['successful_runs']}")
    else:
        print("No success rate could be found, as no run was successful.")

    path = solver.run()
   # solver.plot_maze_path(path)

