#for visualization
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)




#for visualization.
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=data.feature_names,  
                           class_names=data.target_names,  
                           filled=True, rounded=True,  
                           special_characters=True)  
graphviz.Source(dot_data).view() 
