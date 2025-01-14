% Clear output and memory
clc;clear;clc

% Configuration Variables
mu = [0 0]; % Mean of X and Y
Sigma = [2 1.5; 1.5 2]; % Covariance matrix
nSamples = 1000; % Number of samples

% Generate bivariate normal data
data = mvnrnd(mu, Sigma, nSamples);

% Separate X and Y data
x = data(:, 1);
y = data(:, 2);

% Calculate covariance and correlation coefficient
covMatrix = cov(x, y);
covarianceXY = covMatrix(1,2);
correlationXY = corr(x, y);

% Calculate the semivariogram
[variogram, lagDistances] = calculateSemivariogram([x y], 30);

% Create scatter plot
figure;
scatter(x, y, 'filled');
hold on;
title('Bivariate Distribution with 45° Line');
xlabel('X values');
ylabel('Y values');
refline(1, 0); % 45° line
grid on;
axis equal;
legend('Data points', '45° Line');

% Plot the variogram
figure;
plot(lagDistances(1:end-1), variogram, 'o-');
title('Semivariogram');
xlabel('Lag Distance');
ylabel('Semivariance');
grid on;

% Display covariance and correlation
fprintf('Covariance between X and Y: %.3f\n', covarianceXY);
fprintf('Correlation coefficient between X and Y: %.3f\n', correlationXY);