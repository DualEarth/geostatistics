% Define the size of the spatial domain
n = 100; % Number of grid points along each dimension

% Generate a spatial grid
[x, y] = meshgrid(linspace(0, 100, n));

% Define parameters for anisotropic behavior
range_along = 20; % Shorter range of influence along continuity
range_across = 50; % Longer range of influence across continuity
sill = 1; % Variance at the sill
nugget = 0.01; % Small constant added to the diagonal to ensure positive definiteness

% Generate synthetic anisotropic spatial data
% Constructing an anisotropic covariance function
[X1, X2] = meshgrid(x(:));
[Y1, Y2] = meshgrid(y(:));
distance_along = abs(X2 - X1);
distance_across = abs(Y2 - Y1);

% Anisotropic correlation matrices
C_along = sill * exp(-distance_along / range_along);
C_across = sill * exp(-distance_across / range_across);

% Combined covariance matrix
C = (C_along + C_across) / 2;

% Ensuring the matrix is symmetric and positive definite
C = (C + C') / 2 + nugget * eye(n^2); % Adding nugget to the diagonal

% Generate data using Cholesky decomposition
data = reshape(chol(C, 'lower') * randn(n^2, 1), n, n);

% Calculate variograms for different directions
lags = 0:5:n-1;
variogram_along = zeros(size(lags));
variogram_across = zeros(size(lags));

for idx = 1:length(lags)
    lag = lags(idx);
    % Along the direction of continuity
    if lag < n
        variogram_along(idx) = mean((data(:,1:end-lag) - data(:,lag+1:end)).^2, 'all');
    end
    % Across the direction of continuity
    if lag < n
        variogram_across(idx) = mean((data(1:end-lag,:) - data(lag+1:end,:)).^2, 'all');
    end
end

% Plotting the anisotropic variograms
figure;
plot(lags, variogram_along, 'b-o', 'DisplayName', 'Along Continuity');
hold on;
plot(lags, variogram_across, 'r--o', 'DisplayName', 'Across Continuity');
xlabel('Lag Distance');
ylabel('Variogram');
title('Anisotropic Variogram for Different Directions');
legend;
grid on;