import matplotlib.pyplot as plt
import numpy as np

y=np.array([1000,100000, 1000, 100000])
x=np.array(['Donation Money', 'My Money', 'Donation Smartness', 'My Smartness'])
plt.xticks(rotation=45)  # Rotate the x-axis labels by 45 degrees for readability


plt.bar(x,y, color = 'black')
plt.show()