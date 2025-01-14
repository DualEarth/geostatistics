% Clear output and memory
clc;clear;clc

% Configuration Variables
mu = [0 0]; % Mean of X and Y
Sigma = [1 0.5; 0.5 1]; % Initial covariance matrix
nSamples = 500; % Number of samples

% Generate initial bivariate normal data
data = mvnrnd(mu, Sigma, nSamples);

% Separate X and Y data
x = data(:, 1);
y = data(:, 2);

% Define transformations
a = 2; % Scaling factor for Y
b = 3; % Translation factor for X

% Apply transformations
x_translated = x + b;
y_rescaled = a * y;
x_translated_rescaled = x + b;
y_translated_rescaled = a * y;

% Calculate covariance and correlation for original data
covMatrixOriginal = cov(x, y);
correlationOriginal = corr(x, y);

% Calculate covariance and correlation for translated data
covMatrixTranslated = cov(x_translated, y);
correlationTranslated = corr(x_translated, y);

% Calculate covariance and correlation for rescaled data
covMatrixRescaled = cov(x, y_rescaled);
correlationRescaled = corr(x, y_rescaled);

% Calculate covariance and correlation for translated and rescaled data
covMatrixTranslatedRescaled = cov(x_translated_rescaled, y_translated_rescaled);
correlationTranslatedRescaled = corr(x_translated_rescaled, y_translated_rescaled);

% Create scatter plots
figure;
subplot(2,2,1); % Original data
scatter(x, y, 'filled');
title('Original Data');
xlabel('X values');
ylabel('Y values');
axis equal;
grid on;
legend(sprintf('Cov: %.2f, Corr: %.2f', covMatrixOriginal(1,2), correlationOriginal));

subplot(2,2,2); % Translated data
scatter(x_translated, y, 'filled');
title('Translated Data');
xlabel('X translated');
ylabel('Y values');
axis equal;
grid on;
legend(sprintf('Cov: %.2f, Corr: %.2f', covMatrixTranslated(1,2), correlationTranslated));

subplot(2,2,3); % Rescaled data
scatter(x, y_rescaled, 'filled');
title('Rescaled Data');
xlabel('X values');
ylabel('Y rescaled');
axis equal;
grid on;
legend(sprintf('Cov: %.2f, Corr: %.2f', covMatrixRescaled(1,2), correlationRescaled));

subplot(2,2,4); % Translated and Rescaled data
scatter(x_translated_rescaled, y_translated_rescaled, 'filled');
title('Translated & Rescaled Data');
xlabel('X translated');
ylabel('Y rescaled');
axis equal;
grid on;
legend(sprintf('Cov: %.2f, Corr: %.2f', covMatrixTranslatedRescaled(1,2), correlationTranslatedRescaled));

% Display covariance and correlation in the command window for all cases
fprintf('Original Data - Covariance: %.3f, Correlation: %.3f\n', covMatrixOriginal(1,2), correlationOriginal);
fprintf('Translated Data - Covariance: %.3f, Correlation: %.3f\n', covMatrixTranslated(1,2), correlationTranslated);
fprintf('Rescaled Data - Covariance: %.3f, Correlation: %.3f\n', covMatrixRescaled(1,2), correlationRescaled);
fprintf('Translated & Rescaled Data - Covariance: %.3f, Correlation: %.3f\n', covMatrixTranslatedRescaled(1,2), correlationTranslatedRescaled);


% Create scatter plot
figure;
scatter(x, y, 'filled', 'DisplayName', 'Original Data'); hold on;
scatter(x_translated, y, 20, 'filled', 'DisplayName', 'Translated X');
scatter(x, y_rescaled, 20, 'filled', 'DisplayName', 'Rescaled Y');
scatter(x_translated_rescaled, y_translated_rescaled, 20, 'filled', 'DisplayName', 'Translated & Rescaled');
title('Comparison of Transformations on Bivariate Data');
xlabel('X values');
ylabel('Y values');
legend('show');
axis equal;
grid on;

% Display covariance and correlation in the command window for all cases
fprintf('Original Data:\n');
fprintf(' - Covariance: %.3f\n - Correlation: %.3f\n\n', cov(x, y));
fprintf('Translated Data:\n');
fprintf(' - Covariance: %.3f\n - Correlation: %.3f\n\n', cov(x_translated, y));
fprintf('Rescaled Data:\n');
fprintf(' - Covariance: %.3f\n - Correlation: %.3f\n\n', cov(x, y_rescaled));
fprintf('Translated & Rescaled Data:\n');
fprintf(' - Covariance: %.3f\n - Correlation: %.3f\n', cov(x_translated_rescaled, y_translated_rescaled));