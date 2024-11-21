import random
import matplotlib.pyplot as plt
import numpy as np

class Map(object):
    def __init__(self, rows: int, cols: int, start: list, goal: list) -> None:
        if not (isinstance(start, list) and len(start) == 2) or not (isinstance(goal, list) and len(goal) == 2):
            raise Exception("Start and goal must each be a list with exactly 2 elements")
        
        if rows <= start[0] or rows <= goal[0] or cols <= start[1] or cols <= goal[1]:
            raise Exception("Start or goal is out of bounds")
        
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.map = [[0 for _ in range(cols)] for _ in range(rows)]

    def get_map(self):
        return self.map

    def print_map(self):
        for i in self.map:
            print(i)

    def generate_path(self):
        stack = [(self.start[0], self.start[1])] # holds a tuple of the x and y coordinate. initally the start pos is there
        visited = set() # the closed list
        visited.add((self.start[0], self.start[1])) # adding the start to the closed list

        while stack: # while the stack is not empty
            x, y = stack[-1] # get the current cell coordinates
            self.map[x][y] = 1 # mark this cell as part of the path

            if [x, y] == self.goal:
                break  # path complete if we've reached the goal

            # get random possible directions to move (down, up, right, left)
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] 
            random.shuffle(directions) # randomize the movement order
            moved = False

            for dx, dy in directions: # up, down, left, right
                nx, ny = x + dx, y + dy # get the new cell coords
                if (0 <= nx < self.rows and 0 <= ny < self.cols and # ensures the neighbor is within the grid
                        (nx, ny) not in visited and random.random() < 0.7):  # ensure we dont re exapnd already expanded cell. only a 70% chance to explore the neighbor making sure paths are not to dense
                    stack.append((nx, ny)) # append the new cell
                    visited.add((nx, ny)) # add it to the closed list
                    moved = True # a moe was made
                    break

            if not moved:  # if no move was made, backtrack and add dead-end
                stack.pop() # remove the current cell form the backtrack
                if random.random() < 0.3:  # 30% chance to create dead-ends
                    self.map[x][y] = 1  # mark the dead end as a path cell

        # randomly add a few extra dead-ends for complexity
        dead_end_count = random.randint(3, 7)
        for _ in range(dead_end_count):
            x, y = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if self.map[x][y] == 0:
                self.map[x][y] = 1


#map1 = Map(20, 20, [0, 0], [19, 19])
#map1.generate_path()
#map1.print_map()

map2 = np.array([
[-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, -2]
])# -1 = start, -2 = goal


# Define the color map
# -1: start (green), -2: goal (red), 1: path (white), 0: wall(black)
cmap = {
    -1: "green",
    -2: "red",
    1: "white",
    0: "black"
}

# applies the color map to each cell
color_map = np.vectorize(cmap.get)(map2) # basically the cells now contain colors instead of numbers
 
# Plot the map using imshow
plt.figure(figsize=(10, 10))
plt.imshow(map2, cmap='Greys', vmin=-2, vmax=1)  
# this initial call to imshow displays map2 using a grayscale color scheme, ranging from -2 to 1.
# this grayscale background isn’t the final color display but gives a background for positioning cells accurately.

plt.xticks([])  
plt.yticks([])  
#xticks([]) and yticks([]) hide axis ticks for a cleaner look.

# overlay the colors for each cell
for i in range(map2.shape[0]):
    for j in range(map2.shape[1]): # to get every cell
        plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, color=color_map[i, j]))
        # plt.gca() (get current axes) is used to overlay color patches over the grayscale plot.
        # for each cell, plt.Rectangle((j, i), 1, 1, color=color_map[i, j]) creates a square of size 1x1 at position (j, i), with the color defined in color_map[i, j].
        # add_patch adds each rectangle to the plot, covering the grayscale background with the appropriate color.



# show the final visualization
plt.title("Map Visualization")
plt.show()

