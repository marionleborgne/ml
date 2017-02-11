import pandas as pd
import matplotlib.pyplot as plt

# load a csv file to a pandas dataframe
df = pd.read_csv('data/weight-height.csv')

groups = df.groupby('Gender')

# Use pandas to plot
df.plot(kind='scatter', x='Height', y='Weight')
plt.title('Humans')
plt.draw()


# ## Exercises:
# 1) Male / Female population
#    - separate the male and female population by color using
#      df[condition] clauses
#      (http://pandas.pydata.org/pandas-docs/stable/10min.html#boolean-indexing)
#    - label the axes
#    - discuss

plt.figure()
for name, group in groups:
  # group is a DataFrame yay 
  plt.plot(group.Height, group.Weight, 
           marker='o', linestyle='', ms=2, label=name)
plt.legend()
plt.title('Humans by Gender')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

#
# 2) Histograms
#    - plot the histogram of the heights for males and for females
#      on the same plot
#    - use alpha to control transparency
#    - plot a vertical line at the mean using axvline
#
# 3) Check the code in the advanced folder:
#    - advanced/00_pandas_review.ipynb
#    - advanced/01_exploration.ipynb
#    - load any dataset in the data folder
#    - explore it and plot it

# Additional notes and links:
# http://stackoverflow.com/questions/26139423/plot-different-color-for
# -different-categorical-levels-using-matplotlib
