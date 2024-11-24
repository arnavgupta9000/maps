import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import heapq
from queue import Queue
import time
import matplotlib.colors as mcolors
import random


#Map definition (-1 = start, -2 = goal, 1 = open, 0 = wall)
map2 = np.array([
    [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
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
])

# map2 = np.array([
#     [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
# ])

# map2 = np.array([
#     [1,1,1,1,1],
#     [1,1,-1,1,1],
#     [1,1,1,1,1],
#     [1,0,0,0,1],
#     [1,1,-2,1,1]
# ])

# map2 = np.array([
#     [-1,1,1,1,1,1,1,1,1,1],
#     [1,0,0,0,0,0,0,1,0,1],
#     [1,1,1,1,1,0,0,1,0,1],
#     [1,0,0,0,1,1,1,1,0,1],
#     [1,0,0,0,1,0,0,1,0,1],
#     [1,0,0,0,1,1,1,1,0,1],
#     [1,0,0,0,0,0,0,1,0,1],
#     [1,0,0,0,0,0,0,1,1,1],
#     [1,0,0,0,0,0,0,0,0,0],
#     [1,1,1,1,1,1,1,1,1,-2]
# ])

# map2 = np.array([
#     [1,1,1,1,1],
#     [1,1,-2,1,1],
#     [0,0,0,0,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,-1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1],
#     [1,1,0,1,1]
# ])


# Thread-safe updates queue
updates_queue = Queue()

class A_star(object):
    def __init__(self,map_p):
        self.map = map_p
        start = self.find_state('start')
        goal = self.find_state('goal')
        self.max_r, self.max_c = self.num_rows_cols()
        self.grid = []
        self.start = start # (r,c)
        self.goal = goal # (r,c)
    
    def find_state(self, state):
        if state == "start":
            state = -1
        if state == "goal":
            state = -2
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if self.map[i][j] == state:
                    return (i, j)
    
    def num_rows_cols(self):
        return (len(self.map) - 1, len(self.map[0]) - 1)
    
    def generate_neighbhors(self, node):
        # a node will be a a row and a column
        r,c = node # extract the row and the column
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        children = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr <= self.max_r and 0 <= nc <= self.max_c and self.map[nr][nc] != 0 and self.map[nr][nc] != 6:
                children.append((nr, nc))
        
        return children

        
    
    def h_value(self, r, c):
        # ecludian distance
        #return ((r - self.goal[0]) ** 2 + (c - self.goal[1]) ** 2) ** 0.5
        return  abs(r - self.goal[0]) +abs(c-self.goal[1]) 



    
    def run(self):
        
        open = []
        heapq.heappush(open, (0, -0, self.start))  # f-value is initially 0 for the start
        closed = {self.start: 0}
        came_from = {}  # to reconstruct the path

        while len(open) > 0:
            _, _, n = heapq.heappop(open)
            updates_queue.put(("explored", n))
            time.sleep(0.07)

            if n == self.goal:
                self.reconstruct_path(came_from)
                return
            
            cost = closed[n]
            for n_prime in self.generate_neighbhors(n):
                r, c = n_prime
                # Calculate g(n') and f(n') = g(n') + h(n')
                g_n = cost + 1  # g(n') = g(n) + 1
                h_n = self.h_value(r, c)  # heuristic value
                f_n = g_n + h_n  # total estimated cost
                #print(f_n, n_prime)

                
                # Check if the node should be updated in closed or added to open
                if n_prime not in closed or g_n < closed[n_prime]:
                    closed[n_prime] = g_n  # Update g-value
                    came_from[n_prime] = n
                    heapq.heappush(open, (f_n, -g_n, n_prime))  # Use f-value as priority

        return (False, -1)


    def reconstruct_path(self, came_from):
        #print(came_from)
        path = [self.goal] # store the goal

        while True:
            path.append(came_from[path[-1]])
            if path[-1] == self.start: # after appended value
                break
        
        path = path[::-1]
        for i in path:
            updates_queue.put(("path", i))
            time.sleep(0.07)
        print(len(path))
        return



# Start A* simulation in a thread
a_star= A_star(map2)
thread = Thread(target=a_star.run)
thread.start()

class Map_generator(object):

    def __init__(self):
        self.rows = random.randint(5,100)
        self.cols = random.randint(5,100)
        self.prev_h = 0 # store the prev heuristic

        self.map = [[2 for _ in range(self.cols)] for _ in range(self.cols)]
        self.make()


        self.path()


    def make(self):

        self.start = (random.randint(0, self.rows - 1),random.randint(0, self.rows - 1))
        self.goal = (random.randint(0, self.rows - 1),random.randint(0, self.rows - 1))
        if self.start == self.goal:
            self.make()

    def heuristic(self, r,c): 
        # calculate how far the current state is from the goal
        return  abs(r - self.goal[0]) +abs(c-self.goal[1]) 


    
    def path(self):
        # make a path using dfs
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        h_holder = 0 # if this goes below -2 -> move until it hits 0 again, scale by a factor of 0.5
        h_cup = False # toggle to only generate better paths
        found = False
        location = self.start
        while not found:
            if not h_cup:
                choice = directions[random.randint(0,3)] # choose a random way to go
                r,c = choice # get the row and column of the way we moved
                r, c = location[0] + choice[0], location[1] + choice[1] # going to the new location
                self.map[r][c] = 1 # this is a valid path
                h = self.heuristic(r,c)
                if not self.prev_h == 0:
    
                    # see if this h was better than the previous one
                    if h < self.prev_h:
                        h_holder +=1
                    else:
                        # a worse decision was made
                        h_holder -=1

                    if h_holder <= -2:
                        h_cup = True
                self.prev_h = h
                

            else:
                while h_cup < 0:
                    # go the best way to get to the goal by a factor of 2
                    for r,c in directions:
                        h = self.heuristic(r,c)
                        if h < self.prev_h: # good go this way
                            self.map[r][c] = 1
                            h_holder += 0.5
                            self.prev_h = h


            




# Define a custom colormap for your map
#Be in increasing order.
#Cover the entire range of values in your data.
'''
bounds = [-2, -1, 0.5, 0.5, 0.9, 1.5]
[-2, -1) → Maps to the first color (black).
[-1, 0.5) → Maps to the second color (red).
[0.5, 0.5) → This is problematic because it's not a valid interval.
[0.5, 0.9) → Maps to the fourth color (lightblue).
[0.9, 1.5) → Maps to the fifth color (orange).

'''

cmap = mcolors.ListedColormap(['red', 'green', 'black',  'white', 'blue', 'orange'])
bounds = [-2, -1, 0, 1, 4, 10, 14]  # need the 2 to close the bounds
norm = mcolors.BoundaryNorm(bounds, cmap.N)



# Update animation setup
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("A* Pathfinding Visualization")
img = ax.imshow(map2, cmap=cmap, norm=norm)
plt.ion()
plt.show()




def update(frame):
    """Update map visualization based on queue data."""
    while not updates_queue.empty():
        type1, cell = updates_queue.get()
        r, c = cell
        if type1 == 'explored':
            if map2[r][c] != -1 and map2[r][c] != -2:
                map2[r][c] = 6  # blue for explored
        elif type1 == 'path':
            if map2[r][c] != -1 and map2[r][c] != -2:
                map2[r][c] = 12

    img.set_array(map2)
    return [img]

ani = animation.FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)

plt.ioff()
plt.show()
