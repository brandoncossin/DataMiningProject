#import relevant packages
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
#load in the data
df = pd.read_csv('parsed_covid.csv', index_col=0)
#convert to a dataframe
#prints first 5 then the column names
print("* df.head()", df.head(10), sep="\n", end="\n\n")
print(df.columns)

#dummies are "fake" data aka numerically represnting non numeric data
#hot_data = pd.get_dummies(df[['sex', 'age_group', 'race_ethnicity_combined']])
hot_data = pd.get_dummies(df[['sex', 'age_group']])
print("* hot_data.head()", hot_data.head(10), sep="\n", end="\n\n")
print(hot_data.columns)

#feature columns what are independent
#feature_cols = ['sex' , 'age_group', 'race_ethnicity_combined']
#x = df[feature_cols]
#dependent value the target value
#y = df['death_yn']
# training data can be seperated into percentages. some for test and some for training of the model
X_train, X_test, y_train, y_test = train_test_split(hot_data, df['death_yn'], test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(class_weight='balanced')

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)
text_representation = tree.export_text(clf)
print(text_representation)
#plt the figure, setting a black background
plt.figure(figsize = (20, 6), dpi = 80)
#clf.view()
#class names
cn = ['no', 'yes']
fig = tree.plot_tree(clf, 
                   feature_names=hot_data.columns,  
                   #class_names=df['death_yn'],
                   class_names=['unprobable death', 'probable death'],
                   filled=True,
                   rounded=True)
plt.savefig('covid_graph.pdf')
plt.savefig('covid_graph.svg')
plt.show()
