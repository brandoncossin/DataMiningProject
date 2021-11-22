#import relevant packages
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function

#load in the data
df = pd.read_csv('parsed_covid.csv', index_col=0)

#convert to a dataframe
#df = pd.DataFrame(data.data, columns = data.feature_names)
#prints first 5 then the column names
print("* df.head()", df.head(10), sep="\n", end="\n\n")
print(df.columns)

#feature columns what are independent
feature_cols = ['sex' , 'age_group', 'race_ethnicity_combined']
x = df[feature_cols]
#dependent value the target value
y = df['death_yn']
# training data can be seperated into percentages. some for test and some for training of the model
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x, y)
tree.plot_tree(clf)
