clc;clear;clc
% Define the mean and covariance matrix
mu = [0 0]; % mean values of two variables
Sigma = [1 0.8; 0.8 1]; % covariance matrix: variances are 1, covariance is 0.8

% Generate random samples
data = mvnrnd(mu, Sigma, 400); % 1000 samples

% Extract data for plotting
x = data(:, 1);
y = data(:, 2);

% Calculate variance and covariance
var_x = var(x);
var_y = var(y);
cov_xy = cov(x, y); % returns a covariance matrix

% Display results
fprintf('Variance of X: %f\n', var_x);
fprintf('Variance of Y: %f\n', var_y);
fprintf('Covariance matrix:\n');
disp(cov_xy);

% Create a scatter plot
figure;
scatter(x, y, 10, 'filled');
hold on;
title('Bivariate Distribution with Covariance and Variance');
xlabel('X values');
ylabel('Y values');
grid on;
axis equal;

% Add lines to show the variance in the X and Y directions
line([mu(1) - sqrt(var_x), mu(1) + sqrt(var_x)], [mu(2), mu(2)], 'Color', 'r', 'LineWidth', 2); % X variance
line([mu(1), mu(1)], [mu(2) - sqrt(var_y), mu(2) + sqrt(var_y)], 'Color', 'b', 'LineWidth', 2); % Y variance

% Annotate with text
text(mu(1) + sqrt(var_x), mu(2), 'Std Dev in X', 'HorizontalAlignment', 'right');
text(mu(1), mu(2) + sqrt(var_y), 'Std Dev in Y', 'VerticalAlignment', 'bottom');


