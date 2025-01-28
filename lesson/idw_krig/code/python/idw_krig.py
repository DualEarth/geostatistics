import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.linalg import solve
import pandas as pd

np.random.seed(3)  # Set a random seed for reproducibility

def idw_interpolation(obs_coords, obs_values, target_coords, power=1):
    """ Perform IDW interpolation. """
    dists = distance_matrix(target_coords, obs_coords)
    dists[dists == 0] = 1e-10
    weights = 1 / np.power(dists, power)
    weighted_values = weights @ obs_values / weights.sum(axis=1)
    return weighted_values

def compute_semivariogram(Z_coords, Z_values, h_bins):
    """
    Compute experimental semivariogram from spatial data.

    Parameters:
    - Z_coords: coordinates of data points.
    - Z_values: observed values at the data points.
    - h_bins: array of bins to compute the variogram lags.

    Returns:
    - h_lags: middle points of the bins used for lags in variogram.
    - gamma_h: computed semivariogram values for each lag.
    """
    # Ensure h_bins is a numpy array for element-wise operations
    h_bins = np.array(h_bins)

    # Compute pairwise distances between points
    D = distance_matrix(Z_coords, Z_coords)
    # Compute pairwise differences squared
    M = (Z_values[:, None] - Z_values)**2
    # Initialize semivariogram array
    gamma_h = np.zeros(len(h_bins) - 1)

    # Calculate semivariogram for each lag
    for i in range(len(h_bins) - 1):
        mask = (D >= h_bins[i]) & (D < h_bins[i + 1])
        if np.any(mask):
            gamma_h[i] = 0.5 * np.mean(M[mask])

    h_lags = (h_bins[:-1] + h_bins[1:]) / 2
    return h_lags, gamma_h

def simple_kriging(Z_coords, Z_values, target_coords, h_bins):
    """
    Perform simple kriging interpolation to estimate unknown values at target coordinates.

    Parameters:
    - Z_coords: coordinates of known data points.
    - Z_values: values at known data points.
    - target_coords: coordinates where values need to be estimated.
    - h_bins: bins to define the variogram model.

    Returns:
    - Estimates at target coordinates.
    """
    # Compute semivariogram
    h_lags, gamma_h = compute_semivariogram(Z_coords, Z_values, h_bins)

    # Build kriging matrix using variogram values
    A = np.zeros((len(Z_coords) + 1, len(Z_coords) + 1))
    D = distance_matrix(Z_coords, Z_coords)
    A[:-1, :-1] = np.interp(D.flatten(), h_lags, gamma_h).reshape(D.shape)
    A[-1, :] = 1
    A[:, -1] = 1
    A[-1, -1] = 0

    # Matrix of distances to targets
    D_targets = distance_matrix(Z_coords, target_coords)
    B = np.zeros((len(Z_coords) + 1, len(target_coords)))
    B[:-1, :] = np.interp(D_targets.flatten(), h_lags, gamma_h).reshape(D_targets.shape)
    B[-1, :] = 1

    # Solve the kriging system to find weights
    weights = np.linalg.solve(A, B)
    estimates = np.dot(weights[:-1].T, Z_values - np.mean(Z_values))

    return estimates


def kriging(Z_coords, Z_values, target_coords, variogram_model):
    """
    Perform Simple Kriging interpolation using a specified semivariogram model.

    Parameters:
    - Z_coords: Coordinates of known data points.
    - Z_values: Values at known data points.
    - target_coords: Coordinates where values are to be estimated.
    - variogram_model: A function that models the semivariogram.

    Returns:
    - Estimated values at target coordinates.
    """
    n = len(Z_values)
    A = np.zeros((n + 1, n + 1))
    D = distance_matrix(Z_coords, Z_coords)
    A[:n, :n] = variogram_model(D)
    # Ensure kriging matrix is correctly set up with ones on the last row and column
    A[n, :] = 1
    A[:, n] = 1
    A[-1, -1] = 0
    
    # Build matrix B for target locations
    D_target = distance_matrix(Z_coords, target_coords)
    B = np.zeros((n + 1, len(target_coords)))
    B[:n, :] = variogram_model(D_target)
    B[n, :] = 1
    
    # Solve the kriging system to find weights
    weights = np.linalg.solve(A, B)
    # Calculate estimates as a weighted sum of known values
    return weights[:-1].T @ Z_values

def linear_semivariogram(h, slope=1, nugget=0):
    """
    Linear semivariogram model.

    Parameters:
    - h: Distances (lags) at which to calculate semivariogram values.
    - slope: Slope of the linear part of the semivariogram.
    - nugget: Nugget effect at very small distances.

    Returns:
    - Semivariogram values for the given distances.
    """
    return nugget + slope * h

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
xx, yy = np.meshgrid(x, y)
target_coords = np.c_[xx.ravel(), yy.ravel()]

true_raster = np.sin(0.5 * np.sqrt(xx**2 + yy**2)).ravel()

N = 100
obs_coords_indices = np.random.choice(target_coords.shape[0], N, replace=False)
obs_coords = target_coords[obs_coords_indices]
obs_values = true_raster[obs_coords_indices]

unobserved_index = np.random.choice(target_coords.shape[0], 1, replace=False)
unobserved_point = target_coords[unobserved_index]
unobserved_true_value = true_raster[unobserved_index]

idw_values = idw_interpolation(obs_coords, obs_values, target_coords)
kriging_values = kriging(obs_coords, obs_values, target_coords, linear_semivariogram)

unobs_idw = idw_interpolation(obs_coords, obs_values, unobserved_point)
unobs_kriging = kriging(obs_coords, obs_values, unobserved_point, linear_semivariogram)

# Assuming the known mean is the average of the observed values
known_mean = np.mean(obs_values)
# Perform Simple Kriging
simple_kriging_values = simple_kriging(obs_coords, obs_values, target_coords, [i for i in range(4)])
unobs_simple_kriging_values = simple_kriging(obs_coords, obs_values, unobserved_point, [i for i in range(4)])


# Print values at the unobserved location
print(f"IDW value at unobserved point: {unobs_idw[0]}")
print(f"Kriging value at unobserved point: {unobs_kriging[0]}")

# Prepare DataFrame including true values for observed and unobserved points
df = pd.DataFrame({
    'X': np.concatenate((obs_coords[:, 0], unobserved_point[:, 0])),
    'Y': np.concatenate((obs_coords[:, 1], unobserved_point[:, 1])),
    'Z Observed': np.append(obs_values, np.nan),  # "NaN" for unobserved
    'True Value': np.append(obs_values, unobserved_true_value),  # Include true values for all
    'IDW Predicted': np.append(idw_interpolation(obs_coords, obs_values, obs_coords), unobs_idw),
    'Simple Kriging Predicted': np.append(simple_kriging(obs_coords, obs_values, obs_coords, [i for i in range(4)]), unobs_simple_kriging_values),
    'Kriging Predicted': np.append(kriging(obs_coords, obs_values, obs_coords, linear_semivariogram), unobs_kriging)
})

print(df.head())
print(df.tail())


fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Adjusted for an additional plot

# Define color scale limits
vmin = min(np.min(idw_values), np.min(kriging_values), np.min(simple_kriging_values))
vmax = max(np.max(idw_values), np.max(kriging_values), np.max(simple_kriging_values))

# True raster plot
im = axs[0, 0].imshow(true_raster.reshape(100, 100), extent=(0, 10, 0, 10), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axs[0, 0].set_title('True Values')
fig.colorbar(im, ax=axs[0, 0], orientation='vertical')

# IDW plot
sc = axs[0, 1].scatter(xx.ravel(), yy.ravel(), c=idw_values, cmap='viridis', s=9, vmin=vmin, vmax=vmax)
axs[0, 1].scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs_values, edgecolors='black', s=100, cmap='viridis', vmin=vmin, vmax=vmax)
axs[0, 1].scatter(unobserved_point[:, 0], unobserved_point[:, 1], c=unobs_idw, s=100, marker='o', facecolors='none', edgecolors='red', vmin=vmin, vmax=vmax)
axs[0, 1].set_title('IDW Interpolation')
fig.colorbar(sc, ax=axs[0, 1], orientation='vertical')

# Kriging plot
sc = axs[1, 1].scatter(xx.ravel(), yy.ravel(), c=kriging_values, cmap='viridis', s=9, vmin=vmin, vmax=vmax)
axs[1, 1].scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs_values, edgecolors='black', s=100, cmap='viridis', vmin=vmin, vmax=vmax)
axs[1, 1].scatter(unobserved_point[:, 0], unobserved_point[:, 1], c=unobs_kriging, s=100, marker='o', facecolors='none', edgecolors='red', vmin=vmin, vmax=vmax)
axs[1, 1].set_title('Kriging Interpolation')
fig.colorbar(sc, ax=axs[1, 1], orientation='vertical')

# Simple Kriging plot
sc = axs[1, 0].scatter(xx.ravel(), yy.ravel(), c=simple_kriging_values, cmap='viridis', s=9, vmin=vmin, vmax=vmax)
axs[1, 0].scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs_values, edgecolors='black', s=100, cmap='viridis', vmin=vmin, vmax=vmax)
axs[1, 0].scatter(unobserved_point[:, 0], unobserved_point[:, 1], c=unobs_simple_kriging_values, s=100, marker='o', facecolors='none', edgecolors='red', vmin=vmin, vmax=vmax)
axs[1, 0].set_title('Simple Kriging Interpolation')
fig.colorbar(sc, ax=axs[1, 0], orientation='vertical')

plt.tight_layout()
plt.savefig("spatial_interpolation.png")
