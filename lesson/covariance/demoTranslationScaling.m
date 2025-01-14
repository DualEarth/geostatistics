% Clear output and memory
clc;clear;clc

% Configuration Variables
mu = [0 0]; % Mean of X and Y
Sigma = [2 0.5; 0.5 1]; % Initial covariance matrix
nSamples = 1000; % Number of samples

% Generate initial bivariate normal data
data = mvnrnd(mu, Sigma, nSamples);

% Separate X and Y data
x = data(:, 1);
y = data(:, 2);

% Define transformations
a = 2; % Scaling factor
b = 5; % Translation factor

% Apply transformations
x_translated = x + b;
y_rescaled = a * y;

% Calculate covariance and correlation for original data
covMatrixOriginal = cov(x, y);
correlationOriginal = corr(x, y);

% Calculate covariance and correlation for transformed data
covMatrixTransformed = cov(x_translated, y_rescaled);
correlationTransformed = corr(x_translated, y_rescaled);

% Create scatter plots
figure;
subplot(1,2,1); % Original data
scatter(x, y, 'filled');
title('Original Data');
xlabel('X values');
ylabel('Y values');
axis equal;
grid on;
legend(sprintf('Cov: %.2f, Corr: %.2f', covMatrixOriginal(1,2), correlationOriginal));

subplot(1,2,2); % Transformed data
scatter(x_translated, y_rescaled, 'filled');
title('Transformed Data');
xlabel('X translated');
ylabel('Y rescaled');
axis equal;
grid on;
legend(sprintf('Cov: %.2f, Corr: %.2f', covMatrixTransformed(1,2), correlationTransformed));

% Display covariance and correlation in the command window
fprintf('Original Data - Covariance: %.3f, Correlation: %.3f\n', covMatrixOriginal(1,2), correlationOriginal);
fprintf('Transformed Data - Covariance: %.3f, Correlation: %.3f\n', covMatrixTransformed(1,2), correlationTransformed);