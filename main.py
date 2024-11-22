import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import heapq
from queue import Queue
import time
import matplotlib.colors as mcolors


#Map definition (-1 = start, -2 = goal, 1 = open, 0 = wall)
# map2 = np.array([
#     [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, -2]
# ])

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

map2 = np.array([
    [1,1,1,1,1],
    [1,1,-1,1,1],
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,1,-2,1,1]
])



# Thread-safe updates queue
updates_queue = Queue()

def dummy_a_star():
    """Simulates A* search by adding dummy updates."""
    updates_queue.put(("explored", (0, 1)))
    time.sleep(2)
    updates_queue.put(("explored", (2, 2)))
    time.sleep(2)
    updates_queue.put(("explored", (3, 3)))
    time.sleep(2)
    updates_queue.put(("explored", (4, 4)))
    time.sleep(2)
    updates_queue.put(("explored", (5, 5)))
    time.sleep(2)
    #updates_queue.put(("path", [(1,1), (2, 2), (3, 3), (4,4), (5,5)]))

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
    
    def generate_neighbhors(self, node, max_r, max_c):
        # a node will be a a row and a column
        r,c = node # extract the row and the column
        children = []
        if r + 1 <= max_r and self.map[r+1][c] != 0 and self.map[r+1][c] != 6: # already explored
            children.append((r+1,c))
        if c + 1 <= max_c and self.map[r][c+1] != 0 and self.map[r][c+1] != 6: # already explored
            children.append((r, c+1))
        if r - 1 >=0 and self.map[r-1][c] != 0 and self.map[r-1][c] != 6: # already explored
            children.append((r-1, c))
        if c - 1 >= 0 and self.map[r][c-1] != 0 and self.map[r][c-1] != 6: # already explored
            children.append((r, c-1))
        
        return children
    
    def h_value(self, r, c):
        # manhattan distance
        return abs(r - self.goal[0]) + abs(c - self.goal[1])

    def g_value(self, r, c): 
        return abs(r-self.start[0]) + abs(c-self.start[1])

    def f_value(self, r,c):
        return self.g_value(r, c) + self.h_value(r, c)

    
    def run(self):
        open = []
        heapq.heappush(open, (0, self.start))
        closed = {self.start: 0}
        
        while len(open) > 0:

            _, n = heapq.heappop(open)
            updates_queue.put(("explored", n))
            time.sleep(0.1)

            if n == self.goal:
                return (True, closed[self.goal])
            
            
            for n_prime in self.generate_neighbhors(n, self.max_r, self.max_c):
                r,c = n_prime
                # f=g+h
                cost = self.h_value(r,c)
                if n_prime not in closed or cost < closed[n_prime]:
                    closed[n_prime] = closed[n] + 1 # manhattan distance is just +1 for any direction
                    heapq.heappush(open, (cost, n_prime))

        return (False, -1)



# Start A* simulation in a thread
a_star= A_star(map2)
thread = Thread(target=a_star.run)
thread.start()



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

cmap = mcolors.ListedColormap(['red', 'green', 'black',  'white', 'blue'])
bounds = [-2, -1, 0, 1, 4, 10]  # need the 2 to close the bounds
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
        _, cell = updates_queue.get()
        r, c = cell
        if map2[r][c] != -1 and map2[r][c] != -2:
            map2[r][c] = 6  # blue for explored
    img.set_array(map2)
    return [img]

ani = animation.FuncAnimation(fig, update, interval=100, blit=True)

plt.ioff()
plt.show()
