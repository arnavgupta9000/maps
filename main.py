import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import heapq
from queue import Queue
import time
import matplotlib.colors as mcolors


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

map2 = np.array([
    [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
])

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
    
    def generate_neighbhors(self, node, max_r, max_c):
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
            time.sleep(0.03)

            if n == self.goal:
                self.reconstruct_path(came_from)
                return
            
            cost = closed[n]
            for n_prime in self.generate_neighbhors(n, self.max_r, self.max_c):
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
            time.sleep(0.03)
        print(len(path))
        return



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
