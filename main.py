import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
from queue import Queue
import time
import matplotlib.colors as mcolors


# Map definition (-1 = start, -2 = goal, 1 = open, 0 = wall)
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

# Start A* simulation in a thread
thread = Thread(target=dummy_a_star)
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

def update(frame):
    """Update map visualization based on queue data."""
    while not updates_queue.empty():
        update_type, cell = updates_queue.get()
       
        r,c = cell
        map2[r][c] = 6  # Light blue for explored

    img.set_array(map2)

    return [img]

ani = animation.FuncAnimation(fig, update, interval=2000, blit=True)
plt.show()
