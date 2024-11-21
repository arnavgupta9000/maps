import matplotlib.pyplot as plt
import numpy as np
'''
xpoints = np.array([0,1,2,3])
ypoints = np.array([3,8,1,10])
# make sure the 2 axises have the same number of points

#plt.plot(xpoints, ypoints) # if we dont specify x-axis its default will be 0,1,2,3 ...

#plt.plot(xpoints, ypoints, 'o') = plots only the points, ie no line

#plt.plot(xpoints, ypoints, marker = 'o') # wil specify each point with a circle, can also use marker = '*' and a bunch of others

#plt.plot(ypoints, marker = 'o', linestyle = ':', color = 'blue')
# the line above equals this below. use above for more clarity
#plt.plot(ypoints, 'o:b',  ms = 20, mec= 'r', mfc='black') # ms = size of each dot, mec = color border of each point, mfc = the color of the actual circle

# combining everything we can have the flowing example

#plt.plot(ypoints, marker = 'o', linestyle = ':', color = 'b', ms = 5, mec = 'r', mfc = 'black')

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])


plt.plot(x, y)

plt.plot(x[0], y[0], marker='o', color='r', ms= 4)  # Marker at the beginning
plt.plot(x[-1], y[-1], marker='o', color='r', ms = 4)  # Marker at the end

# for labels and title
plt.xlabel('x-label')
plt.ylabel('y-label')
plt.title('title ')

# this line will add grid lines to the plot
plt.grid() # can specify axis and like color, linestyle, linewidth


plt.show()
'''

'''
# looking at subplots now

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1) # it goes row, column, index therefore this has 1 row, 2 columns, and is index 1
plt.plot(x,y)

plt.title('graph 1')
plt.xlabel('x-axis-1')
plt.ylabel('y-axis-1')

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1,2,2)
plt.plot(x,y)
#plt.subplot(1, 2, 1) (1 row, 2 columns): First subplot on the left, second on the right (ordered by columns).
#plt.subplot(2, 1, 1) (2 rows, 1 column): First subplot on the top, second on the bottom (ordered by rows).

#x = np.array([0, 1, 2, 3])
#y = np.array([1, 12, 22, 33])
#plt.subplot(1,2,3) 
#plt.plot(x,y) 
# this will now error out. think of it like a grid wtih r*c total spots. we have 2 spots with 1 * 2 and if we add a third spot it wont work. but if just change the subplot to be subplot(2,2, x) it will now show it in the top left, right then bottom left, leaving the bottom right space open
plt.suptitle('this will be the overarching title')
plt.show()

'''

# bar graphs
'''
x = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
y = np.array([2,4,8,16,32,62])

plt.bar(x,y, color = 'black', width = 0.8) # bo barh for horizontal bar, for barh use 'height' instead of 'width'
plt.show()

'''

import matplotlib.pyplot as plt
import numpy as np
import random


# Generate random arrays as described
array_1 = [random.randint(0, 60) for _ in range(random.randint(0, 100))]
array_2 = [random.randint(0, 60) for _ in range(random.randint(0, 100))]
array_3 = [random.randint(0, 60) for _ in range(random.randint(0, 100))]
array_4 = [random.randint(0, 60) for _ in range(random.randint(0, 100))]
array_5 = [random.randint(0, 60) for _ in range(random.randint(0, 100))]

# List of the arrays
arrays = [array_1, array_2, array_3, array_4, array_5]

# Step 1: Calculate the average of each array, avoiding empty arrays
averages = [np.mean(arr) if len(arr) > 0 else 0 for arr in arrays]

# Step 2: Create the bar chart
x = np.arange(len(averages))  # X positions for bars
plt.bar(x, averages, color='skyblue', label='Average')

# Step 3: Add dots on top of each bar
plt.scatter(x, averages, color='red', zorder=5)  # Dots at the top of the bars

# Step 4: Add a line connecting the tops of the bars
plt.plot(x, averages, color='green', marker='o', linestyle='-', zorder=10)  # Line with markers

# Adding labels and title
plt.xlabel('Arrays')
plt.ylabel('Average Value')
plt.title('Bar Chart with Average Line and Dots')
plt.xticks(x, [f'Array {i+1}' for i in range(len(arrays))])  # Custom X-tick labels
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()