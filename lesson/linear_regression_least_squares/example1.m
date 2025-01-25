% Define the dataset
x = [1; 2; 3; 4; 5]; % Predictor values
y = [2; 4; 5; 4; 5]; % Response values

% Number of data points
N = length(y);

% Step 1: Construct the design matrix X with a column of ones for intercept
X = [x, ones(N, 1)];  % Design matrix includes column for 'b' and 'x' for 'a'

% Step 2: Compute elements for matrix M
% These calculations correspond to summing x^n, x*y, y, and x^2 over all data points
Sum_x2 = sum(x.^2);  % Sum of x squared
Sum_x = sum(x);      % Sum of x
Sum_xy = sum(x .* y);  % Sum of x*y
Sum_y = sum(y);      % Sum of y

% Matrix M and vector v from the equations derived by setting partial derivatives to zero
M = [Sum_x2, Sum_x; Sum_x, N];
v = [Sum_xy; Sum_y];

% Step 3: Solve for coefficients [a; b] using the normal equations
% The solution is obtained by solving the matrix equation M*[a; b] = v
coefficients = M \ v;  % Solves the linear system

% Extract coefficients
a = coefficients(1);  % Slope
b = coefficients(2);  % Intercept

% Display results
fprintf('The slope (a) is: %.2f\n', a);
fprintf('The intercept (b) is: %.2f\n', b);

% Step 4: Plot the data and the fitted line
y_fitted = X * coefficients;  % Calculate fitted values

figure;
plot(x, y, 'ko', 'MarkerFaceColor', 'k');  % Plot the original data points
hold on;
plot(x, y_fitted, 'b-', 'LineWidth', 2);   % Plot the fitted line
xlabel('X (Predictor)');
ylabel('Y (Response)');
title('Linear Least Squares Regression');
legend('Data Points', 'Fitted Line', 'Location', 'best');
grid on;