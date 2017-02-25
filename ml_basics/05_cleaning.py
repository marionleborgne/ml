import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
input_file_path = os.path.join('data', 'titanic-train.csv')
df = pd.read_csv(input_file_path, index_col=None, header=0)


# Create new features or dummy variables
# drop 2 cases missing on Embarked
df = df[df['Embarked'].notnull()]

# change fare to integer
df['Fare2'] = df['Fare'].astype(int)

# dummy var for Gender == male {male: 1, female: 0}
df['Male'] = 0

df.loc[df['Sex'] == 'male', 'Male'] = 1


# change pclass to categorical variables
# Note: not necessary to apply .get_dummies() but interesting to see a map()
df['Pclass2'] = df.Pclass.map({1: 'one', 2: 'two', 3: 'three'})

# create dummy variables for Pclass2
dum1 = pd.get_dummies(df['Pclass2'], prefix='Pclass2')
df = df.join(dum1)

# fill missing age data with random normalized age data based
# on existing data's mean and std
seed = np.random.RandomState(1)
df['Age2'] = df['Age'].apply(lambda x: seed.normal(df.Age.mean(),
                                                   df.Age.std())
                             if np.isnan(x) else x)
df['Age2'] = df['Age2'].astype(int)


print 'Imputed Age Mean', df['Age2'][df['Age'].isnull()].mean()
print 'Imputed STD', df['Age2'][df['Age'].isnull()].std()
print
print 'Actual Age Mean', df['Age'].mean()
print 'Actual STD', df['Age'].std()
print

# Inspect the distribution of Fare2
a = df['Fare2'].hist(bins=50)
plt.title('Fare2 Histogram')
plt.show(a)

# Count how many passengers paid between 10 and 50?
print df.loc[(df.Fare2 <= 50) & (df.Fare2 >= 10), 'Fare2'].value_counts()

p10 = df['Fare2'][(df.Fare2 <= 10)].count()
print 'Number of Passengers with Fare <=10:', p10

p11_50 = df['Fare2'][(df.Fare2 <= 50) & (df.Fare2 > 10)].count()
print 'Number of Passengers with Fare 11-50:', p11_50

p51 = df['Fare2'][(df.Fare2 > 50)].count()
print 'Number of Passengers with Fare 51+:', p51

# Create new dummy variables for Fare2 according to values in the histogram
df['Fare3'] = ''
df.loc[(df.Fare2 <= 10), 'Fare3'] = 'Fare<=10'
df.loc[(df.Fare2 <= 50) & (df.Fare2 > 10), 'Fare3'] = 'Fare11to50'
df.loc[(df.Fare2 > 50), 'Fare3'] = 'Fare51+'
dum3 = pd.get_dummies(df['Fare3'], prefix='Fare3')
df = df.join(dum3)


df['Family'] = 0
# Note: use '|' instead of 'or'
df.loc[(df.Parch > 0) | (df.SibSp > 0), 'Family'] = 1 

dum4 = pd.get_dummies(df.Embarked, prefix='Embarked')
df.join(dum4)

outfile_path = os.path.join('data','my_new_features.csv')
df.to_csv(outfile_path)

infile_path = os.path.join('data','my_new_features.csv')
original_df = pd.read_csv(input_file_path)


# Exercises
#
# 1) OK create a new feature for the presence of Family combining the 
# information
#    present in SibSp and in ParCh. If a person has a SibSp or a Parch then
#    he/she has a Family
#
# 2) OK create dummy variables for embarked using the get_dummies function
#    as shown above
#
# 3) OK save the final df to a csv file named my_new_features.csv inside
#    the data folder using the to_csv() method
#
# 4) compare with the new_titanic_features.csv file
#
# 5) use some of these techniques on another file
#
# 6) Check the code in the advanced folder:
#    05_cleaning.ipynb
