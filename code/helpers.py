import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import string
import cmocean

# 3D plotting helper
def plot_mse_3D(w1_vec, w2_vec, mse):
    # Obtain the index of the best weights
    w_index = np.array(np.unravel_index(mse.argmin(), mse.shape))

    fig = go.Figure(
        data=[
            go.Surface(z=mse, x=w1_vec, y=w2_vec, opacity=0.9),
            go.Scatter3d(x=[w1_vec[w_index[0]]], y=[w2_vec[w_index[1]]], 
                         z=[mse.min()])
        ]
    )
    fig.update_layout(title='Mean Squared Error Surface', height=600, width=800, 
                      autosize=False,
                      scene=dict(xaxis_title="w1", yaxis_title="w2", 
                                 zaxis_title="MSE")
    )
    fig.show()
    
# Correlation plot helper
def corrplot(df, features, color_features, shape_features=None, figsize=(12, 12)):
    markers = ["o", "^", "s", "P", "*", "H", "X"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    p = len(features)
    # Create the canvas
    fig, axs = plt.subplots(p, p, figsize=figsize)
    for i in range(p):
        for j in range(p):
            ax = axs[i, j]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            if i == j:
                ax.annotate(features[i], (0.5, 0.5), ha="center", va="center")    
            else:
                #ax.grid(linestyle="dashdot")
                for c,f in enumerate(df[color_features].unique()):
                    if shape_features:
                        for k,m in enumerate(df[shape_features].unique()):
                            dft = df[(df[color_features] == f) & (df[shape_features] == m)]
                            if i == 0 and j == 1:
                                ax.scatter(dft[features[i]], dft[features[j]], label=f"{f} ({m})",
                                           alpha=0.8, marker=markers[k], color=colors[c])
                            else:
                                ax.scatter(dft[features[i]], dft[features[j]], marker=markers[k], alpha=0.8, color=colors[c])
                    else:
                        dft = df[df[color_features] == f]
                        if i == 0 and j == 1:
                            ax.scatter(dft[features[i]], dft[features[j]], label=f, alpha=0.8)
                        else:
                            ax.scatter(dft[features[i]], dft[features[j]], alpha=0.8)
                if i == 0: #
                    if j % 2 == 1:
                        ax.xaxis.set_visible(True)
                        ax.xaxis.tick_top()
                elif i == p - 1:
                    if j % 2 == 0:
                        ax.xaxis.set_visible(True)
                ###
                if j == 0:
                    if i % 2 == 1:
                        ax.yaxis.set_visible(True)
                elif j == p - 1:
                    if i % 2 == 0:
                        ax.yaxis.set_visible(True)
                        ax.yaxis.tick_right()
    fig.legend()
    fig.suptitle("Correlation Plot")
    

def plot_kmeans(kmeans, X, h=0.01):
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.imshow(
        Z, 
        interpolation="nearest", 
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=cmocean.cm.haline,
        aspect="auto",
        origin="lower",
        alpha=0.5
    )
    # Plot the scatter
    ax.scatter(X[:, 0], X[:, 1], color="black", edgecolor="white")
    # Plot the centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=180, linewidth=1, color="black", edgecolor="white")
    plt.show()
    

def custom_plot_kmeans(kmeans, df, h=.01, figsize=(12, 8), title="", save=True):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, ax = plt.subplots(figsize=figsize)
    
    xvar, yvar = "sepal length (cm)", "sepal width (cm)"
    
    X = np.array(df[[xvar, yvar]])
    kmeans.fit(X)
    
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
   
    # Obtain labels for each point in mesh
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.imshow(
        Z, 
        interpolation="nearest", 
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=cmocean.cm.haline,
        aspect="auto",
        origin="lower",
        alpha=.5
    )
    for i, species in enumerate(df["species"].unique()):
        # Plot the scatter
        ax.scatter(
            df.loc[df["species"] == species, xvar],
            df.loc[df["species"] == species, yvar], 
            color=colors[i], label=species, edgecolor="white")
    # Plot the centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=180, 
               linewidth=1, color="black", edgecolor="white", label="Cluster centroid")
    ax.legend()
    ax.set_title(title)
    if save:
        plt.savefig(f"../data/images/{max_itr}.png")
    else:
        plt.show()
    
# ===== Code checker for exercises =============================================
# 06a, Task 1
def check_bmi_function(f):
    n_tests = 1000
    heights = np.random.randint(150, 210, n_tests)
    weights = 70 + np.random.rand(n_tests) * 20
    
    def bmi_category(height_in_cm, weight_in_kg):
        bmi = weight_in_kg / (height_in_cm / 100) ** 2
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        return cat
    
    if all([f(h, w) == bmi_category(h, w) for (h, w) in zip(heights, weights)]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")

# 06b, Task 2
def check_town_canton_extractor(f):
    n_tests = 1000
    abc = np.array(list(string.ascii_lowercase))
    strings = [
        "".join(np.random.choice(abc, np.random.randint(1, 20)))
        + " " + "(" + "".join(np.random.choice(abc, np.random.randint(1, 20))) 
        + ")" for _ in range(n_tests)]

    def extract_town_canton(input_string):
        # Use .split to separate the town and canton
        town, canton = input_string.split(" ")
        canton = canton.strip("()")
        return town, canton # Output results
    
    if all([f(s) == extract_town_canton(s) for s in strings]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")
    
# 06c, Task 1
def check_bin_returns(f):
    n_tests = 1000
    returns = np.random.rand(n_tests) * 20
    
    def bin_returns(x):
        # Create the string for the sign (positive/negative)
        sgn = "positive " if x > 0 else "negative "
        if abs(x) > 5: # Extreme returns
            adj = "extreme "
        elif abs(x) > 2: # Large returns
            adj = "large "
        else: # 'Normal' returns, adjective is blank
            adj = ""
        # Return the classification of returns
        return adj + sgn + "returns"
    
    if all([f(r) == bin_returns(r) for r in returns]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")
        
# 113 SVD, Task 1
def check_truncated_svd(f):
    n_tests = 1000
    matrices = [np.random.rand(1 + np.random.randint(100), 
                               1 + np.random.randint(100)) for _ in range(n_tests)]
    ranks = [1 + np.random.randint(np.linalg.matrix_rank(A)) for A in matrices]
    
    
    def truncated_svd(A, k):
        # Perform SVD
        U, S, V = np.linalg.svd(A, full_matrices=False)
        # Truncate and return the matrices
        return U[:, :k], np.diag(S[:k]), V[:k, :]
    
    if all([all([np.all(x == y) for x, y in zip(f(A, k), truncated_svd(A, k))]) for (A, k) in zip(matrices, ranks)]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")
    