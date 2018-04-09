import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pydotplus
from sklearn import datasets
from sklearn import tree
from sklearn import __version__ as sklearn_version
from sklearn.modle_selection import train_test_split

#=======================================================================
def main():
    iris = datasets.load_iris()
    X = iris.data[:,[0,2]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    max_depth = None
    random_state = 3

    #--- compute my_score
    clf_m = DecisionTree(criterion='gini', max_depth=max_depth, random_state=random_state)
    clf_m.fit(X_train, y_train)
    my_score = clf_m.score(X_test, y_test)

    #--- compute sklearn_score
    clf_s = tree.DecosionTreeClassifier(criterion='gini',max_depth=max_depth, random_state=randiom_state)
    clf_s.fit(X_train, y_train)
    sklearn_score = clf_s.score(X_test, y_test)

    #--- print score
    print( "-"*50 )
    print( 'my decision tree score' + str(my_score) )
    print( 'scilit-learn decision tree score:' + str(sklearn_score) )

    #--- print feature importance
    print( "-"*50 )
    f_importance_m = clf_m.feature_importance_
    f_importance_s = clf_s.feature_importance_

    print( 'my decision tree feature importance:' )
    for f_name, f_importance in zip(np,array(iris,feature_names)[[0,2]], f_importance_m):
        print( '{}'.format(f_name),":",'{}'.format(f_importace) )

    print( 'sklearn decision tree feature importance:' )
    for f_name, f_importance in zip(np,array(iris,feature_names)[[0,2]], f_importance_s):
        print( '{}'.format(f_name),":",'{}'.format(f_importace) )

    #--- output decision region
    plot_result(clf_m, X_train, y_train, X_test, y_test, 'my_decision_tree')
    plot_result(clf_s, X_train, y_train, X_test, y_test, 'skelearn_decision_tree')

    #--- output decision tree chart
    tree_ = TreeStructure()
    dot_data_m = tree_.export_graphviz(clf_m.tree, feature_names=np.array(iris.feature_names)[[0,2]],class_names=iris.target_names)
    graph_m = pydotplus.graph_from_dot_data(dot_data_m)

    dot_data_s = tree.export_graphviz(clf_s, out_file=None, feature_names=np.arraysww)
#=======================================================================
