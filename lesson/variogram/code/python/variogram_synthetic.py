import numpy as np
import matplotlib.pyplot as plt

def generate_data(n, condition):
    x = np.linspace(0, 500, n)
    if condition == 1:  # Autocorrelated data decreasing with distance
        z = np.sin(x/50) + np.random.normal(0, 0.5, n)
    elif condition == 2:  # Non-autocorrelated data
        z = np.random.normal(0, 1, n)
    elif condition == 3:  # Autocorrelation with periodic reappearance
        z = np.sin(x/50) + np.sin(x/350) + np.random.normal(0, 0.5, n)
    return x, z

def semivariogram(x, z):
    n = len(x)
    max_distance = np.max(x) - np.min(x)
    distances = np.linspace(0, max_distance, 50)  # Reduced number of distance bins for speed
    variances = []
    for d in distances:
        diffs = []
        for i in range(n):
            for j in range(n):
                if abs(x[i] - x[j]) <= d:
                    diffs.append((z[i] - z[j])**2)
        variances.append(np.mean(diffs))
    return distances, variances

# Generate data for each condition
x1, z1 = generate_data(360, 1)
x2, z2 = generate_data(360, 2)
x3, z3 = generate_data(360, 3)

# Compute semivariograms
dist1, var1 = semivariogram(x1, z1)
dist2, var2 = semivariogram(x2, z2)
dist3, var3 = semivariogram(x3, z3)

# Plotting and saving to files
plt.figure(figsize=(4, 3))
plt.plot(dist1, var1, lw=5)
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.tight_layout()
plt.savefig("semivariogram_condition1.png")

plt.figure(figsize=(4, 3))
plt.plot(dist2, var2, lw=5)
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.tight_layout()
plt.savefig("semivariogram_condition2.png")

plt.figure(figsize=(4, 3))
plt.plot(dist3, var3, lw=5)
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.tight_layout()
plt.savefig("semivariogram_condition3.png")