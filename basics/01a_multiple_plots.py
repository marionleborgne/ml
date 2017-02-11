import numpy as np
import matplotlib.pyplot as plt

# generate some random data
normal_dist = np.random.normal(0, 0.01, 1000)

# init figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(2, 2, 1)
ax.text(0, 0.025, 'Some example text', fontsize=10)
ax.plot(normal_dist)
ax.set_title('Line plot', size=10)
ax.axhline(linewidth=10, color='#d62728', alpha=0.5)
ax.axvspan(800, 900, facecolor='#2ca02c', alpha=0.5)

ax = fig.add_subplot(2, 2, 2)
ax.plot(normal_dist, 'x', color='r')
ax.set_title('Scatter plot', size=10)

ax = fig.add_subplot(2, 2, 3)
ax.hist(normal_dist, bins=50, color='g')
ax.set_title('Histogram', size=10)
ax.set_xlabel('count', size=10)

ax = fig.add_subplot(2, 2, 4)
ax.boxplot(normal_dist)
ax.set_title('Boxplot', size=10)
ax.text(0.5, 0.03, 'Boxes are cool', fontsize=10)

plt.draw()
plt.show()

# Exercises
# - OK change marker type to a square in the scatter plot
# - OK try other letters and symbols
# - OK change color of the histogram to green
# - OK see if you can change the color of the Scatter Plot to red
# - OK add text to the plot using the ax.text() method
#   (doc here: http://matplotlib.org/users/text_intro.html)
# - OK see if you can write "Boxes are cool" in the boxplot?
# - OKadd vertical and horizontal lines with plt.axhline() and plt.axvline()
#   (doc here: http://matplotlib.org/examples/pylab_examples/axhspan_demo.html)
