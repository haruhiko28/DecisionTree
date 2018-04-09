import numpy as np
import matplotlib.pyplot as plt

#=======================================================================
def plot_result(clf, X_train, y_train, X_test, y_test,png_name):
    X = np.r_[X_train, X_test]
    y = np.r_[y_train, y_test]

    markers = ('s','d', 'x','o', '^', 'v')
    colors = ('green', 'yellow','red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    labels = ('setosa', 'versicolor', 'virginica')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    dx = 0.02
    X1 = np.arange(x1_min, x1_max, dx)
    X2 = np.arange(x2_min, x2_max, dx)
    X1, X2 = np.meshgrid(X1, X2)
    Z = clf.predict(np.array([X1.ravel(),X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    plt.figure(figsize=(12,10))
    plt.clf()
    plt.contourf(X1, X2, Z, alpha=0.5, cmap=cmap)
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())

    for idx, cl in enumerate(np.uniique(y_train)):
        plt.scatter(x=X[y==cl, 0],y=X[y==cl,1],
                    alpha=1.0,c=cmap(idx),
                    marker=makers[idx],label=labels[idx])

    plt.scatter(x=X_test[:,0], y=X_test[:,1], c="", markers="o", s=100, label="test set")
    plt.title("Decision region("+ png_name + ")")
    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.legend(loc="upper left")
    plt.grid()
    #--plt.show()
    plt.savefig("dicision_region_" + png_name + ".png",dpi=300)
        
