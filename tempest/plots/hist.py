import numpy as np
import matplotlib.pyplot as plt


def simple_hist(X):
    X_mean = np.mean(X)
    X_std = np.std(X)

    # Create histogram with manually adjusted bins
    bins = int(np.sqrt(len(X)))

    plt.hist(X, bins=bins, edgecolor='black', alpha=0.7)

    # Add mean and standard deviation to the plot
    plt.axvline(X_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {X_mean:.2f}')
    plt.axvline(X_mean + X_std, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {X_std:.2f}')
    plt.axvline(X_mean - X_std, color='green', linestyle='dashed', linewidth=2)

    # Add labels and title
    plt.xlabel('mean qv_sat')
    plt.ylabel('Frequency')
    plt.title('Histogram of mean qv_sat with Mean and Std Dev, Bounds to 0')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()