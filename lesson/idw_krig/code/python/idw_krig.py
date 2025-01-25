import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.linalg import solve
import pandas as pd


# Set a random seed for reproducibility
np.random.seed(3)

# Function to perform IDW interpolation
def idw_interpolation(obs_coords, obs_values, target_coords, power=1):
    """
    Perform IDW interpolation.
    :param obs_coords: Coordinates of observations (Nx2 numpy array).
    :param obs_values: Observed values (N-element array).
    :param target_coords: Coordinates where to interpolate (Mx2 numpy array).
    :param power: Power parameter of the IDW.
    :return: Interpolated values at target coordinates.
    """
    # Calculate the distance matrix
    dists = distance_matrix(target_coords, obs_coords)
    # Avoid division by zero
    dists[dists == 0] = 1e-10
    weights = 1 / np.power(dists, power)
    weighted_values = weights @ obs_values / weights.sum(axis=1)
    return weighted_values

# Function to perform Kriging interpolation
def kriging(obs_coords, obs_values, target_coords, variogram_model):
    """
    Perform Kriging interpolation using a given variogram model.
    :param obs_coords: Coordinates of observations (Nx2 numpy array).
    :param obs_values: Observed values (N-element array).
    :param target_coords: Coordinates where to interpolate (Mx2 numpy array).
    :param variogram_model: A function that defines the variogram model.
    :return: Interpolated values at target coordinates.
    """
    # Number of observations
    n = len(obs_values)
    # Matrix A for Kriging system: Covariance between observed points
    A = np.zeros((n+1, n+1))
    A[:n, :n] = variogram_model(distance_matrix(obs_coords, obs_coords))
    A[n, :] = 1
    A[:, n] = 1
    A[-1, -1] = 0  # Lagrange multiplier for unbiasedness
    
    # Vector B for Kriging system: Covariance between observed points and target points
    B = np.zeros((n+1, len(target_coords)))
    B[:n, :] = variogram_model(distance_matrix(obs_coords, target_coords))
    B[n, :] = 1
    
    # Solve the Kriging system
    weights = solve(A, B)
    return weights[:-1].T @ obs_values

# Variogram model: Simple spherical model
def spherical_variogram(distances, range=10, sill=1):
    """
    Spherical variogram model.
    :param distances: Distances matrix or array.
    :param range: Range parameter of the variogram.
    :param sill: Sill parameter of the variogram.
    :return: Variogram values for the given distances.
    """
    return sill * (1.5 * (distances / range) - 0.5 * (distances / range)**3) * (distances <= range) + sill * (distances > range)

# Define target grid (for visualization and as the domain for true values)
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
xx, yy = np.meshgrid(x, y)
target_coords = np.c_[xx.ravel(), yy.ravel()]

# Generating a synthetic "true values" raster over the meshgrid
# For simplicity, let's assume it's a function of x and y, for example, a peak function
true_raster = np.sin(0.5 * np.sqrt(xx**2 + yy**2)).ravel()

# Generate random observation data
N = 100  # Number of observations
obs_coords_indices = np.random.choice(target_coords.shape[0], N, replace=False)  # Random indices for observations
obs_coords = target_coords[obs_coords_indices]  # Observation coordinates
obs_values = true_raster[obs_coords_indices]  # Observation values from the true raster

# Define an unobserved location and get its true value from the raster
unobserved_index = np.random.choice(target_coords.shape[0], 1, replace=False)  # Random index for unobserved point
unobserved_point = target_coords[unobserved_index]  # Unobserved coordinates
unobserved_true_value = true_raster[unobserved_index]  # True value at unobserved location

# Perform interpolations
idw_values = idw_interpolation(obs_coords, obs_values, target_coords)
kriging_values = kriging(obs_coords, obs_values, target_coords, spherical_variogram)

# Predict values at the unobserved location
unobs_idw = idw_interpolation(obs_coords, obs_values, unobserved_point)
unobs_kriging = kriging(obs_coords, obs_values, unobserved_point, spherical_variogram)

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
    'Kriging Predicted': np.append(kriging(obs_coords, obs_values, obs_coords, spherical_variogram), unobs_kriging)
})

print(df.head())
print(df.tail())

# Ensure that 'Z Observed' is float and handle NaN for plotting
df['Z Observed'] = pd.to_numeric(df['Z Observed'], errors='coerce')

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjust subplot to include a third plot for true values

# True raster plot adjusted to match the scatter plots' visual representation
true_raster_plot = axs[0].imshow(true_raster.reshape(100, 100), extent=(0, 10, 0, 10), origin='lower', cmap='viridis', aspect='auto')
axs[0].set_title('True Values')
fig.colorbar(true_raster_plot, ax=axs[0], orientation='vertical')

# IDW plot
idw_plot = axs[1].scatter(xx.ravel(), yy.ravel(), c=idw_values, cmap='viridis', norm=plt.Normalize(vmin=true_raster.min(), vmax=true_raster.max()), s=9)  # Set s=1 for scatter point size
axs[1].scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs_values, edgecolors='black', s=100, cmap='viridis', norm=plt.Normalize(vmin=true_raster.min(), vmax=true_raster.max()))
axs[1].scatter(unobserved_point[:, 0], unobserved_point[:, 1], c=unobs_idw, s=100, marker='o', facecolors='none', edgecolors='red')  # Use filled circle with facecolor 'none'
axs[1].set_title('IDW Interpolation')
fig.colorbar(idw_plot, ax=axs[1], orientation='vertical')

# Kriging plot
kriging_plot = axs[2].scatter(xx.ravel(), yy.ravel(), c=kriging_values, cmap='viridis', norm=plt.Normalize(vmin=true_raster.min(), vmax=true_raster.max()), s=9)  # Set s=1 for scatter point size
axs[2].scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs_values, edgecolors='black', s=100, cmap='viridis', norm=plt.Normalize(vmin=true_raster.min(), vmax=true_raster.max()))
axs[2].scatter(unobserved_point[:, 0], unobserved_point[:, 1], c=unobs_kriging, s=100, marker='o', facecolors='none', edgecolors='red')  # Use filled circle with facecolor 'none'
axs[2].set_title('Kriging Interpolation')
fig.colorbar(kriging_plot, ax=axs[2], orientation='vertical')

plt.tight_layout()
plt.show()