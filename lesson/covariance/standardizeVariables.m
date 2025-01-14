% Clear output and memory
clc;clear;clc
% Generate synthetic data for X and Y
rng('default'); % For reproducibility
nSamples = 500;
X = randn(nSamples, 1) * 2 + 10; % Mean 10, SD 2
Y = 0.8 * X + randn(nSamples, 1) * 1 + 5; % Correlated with X

% Calculate mean and standard deviation before transformation
mx = mean(X);
my = mean(Y);
ax = std(X);
ay = std(Y);

% Standardize variables (Z-score normalization)
X_prime = (X - mx) / ax;
Y_prime = (Y - my) / ay;

% Calculate covariance, variance, and correlation before and after standardization
covMatrixOriginal = cov(X, Y);
covarianceOriginal = covMatrixOriginal(1,2);
varianceXOriginal = covMatrixOriginal(1,1);
varianceYOriginal = covMatrixOriginal(2,2);
correlationOriginal = corr(X, Y);

covMatrixStandardized = cov(X_prime, Y_prime);
covarianceStandardized = covMatrixStandardized(1,2);
varianceXStandardized = covMatrixStandardized(1,1);
varianceYStandardized = covMatrixStandardized(2,2);
correlationStandardized = corr(X_prime, Y_prime);

% Create scatter plot to visualize original and standardized data
figure;
subplot(1,2,1);
scatter(X, Y, 'filled');
title('Original Data');
xlabel('X values');
ylabel('Y values');
axis equal;
grid on;

subplot(1,2,2);
scatter(X_prime, Y_prime, 'filled');
title('Standardized Data');
xlabel('Standardized X');
ylabel('Standardized Y');
axis equal;
grid on;

% Prepare table
VarNames = {'Variance_X', 'Variance_Y', 'Covariance', 'Correlation'};
RowNames = {'Original', 'Standardized'};
Data = [varianceXOriginal, varianceYOriginal, covarianceOriginal, correlationOriginal;
        varianceXStandardized, varianceYStandardized, covarianceStandardized, correlationStandardized];

% Create table
resultTable = table(Data(:,1), Data(:,2), Data(:,3), Data(:,4), 'VariableNames', VarNames, 'RowNames', RowNames);

% Display table
disp(resultTable);