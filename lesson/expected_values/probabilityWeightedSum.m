% Example random variable values and their corresponding probabilities
values = 0:5; % Discrete values
probabilities = [0.1, 0.15, 0.20, 0.25, 0.20, 0.10]; % Probabilities for each value

% Calculate expected value
expected_value = sum(values .* probabilities);

% Calculate contribution of each value to the expected value
contributions = values .* probabilities;

% Calculate cumulative sum of contributions
cumulative_sums = cumsum(contributions);

% Create a figure to visualize the probabilities and their contributions to the expected value
figure;
bar(values, probabilities, 'b', 'FaceAlpha', 0.6);
title('Probability Distribution and Contributions to Expected Value');
xlabel('Random Variable Values');
ylabel('Probability');
grid on;

% Annotate contributions on the bar chart
for i = 1:length(values)
    text(values(i), probabilities(i), sprintf('  +%.2f', contributions(i)), ...
         'VerticalAlignment', 'bottom', 'FontSize', 10, 'Color', 'red');
end

% Mark expected value
line([expected_value, expected_value], [0, max(probabilities)], 'Color', 'red', 'LineStyle', '--');
text(expected_value, max(probabilities) * 0.9, sprintf('Expected Value = %.2f', expected_value), ...
     'HorizontalAlignment', 'center', 'Color', 'red', 'FontSize', 10);

% Print the table to the console
fprintf('\nTable of Values, Probabilities, Contributions, and Cumulative Sum:\n');
fprintf('-----------------------------------------------------------------------------\n');
fprintf('%-6s %-12s %-15s %-15s\n', 'Value', 'Probability', 'Contribution', 'Cum. Sum');
fprintf('-----------------------------------------------------------------------------\n');
for i = 1:length(values)
    fprintf('%-6d %-12.2f %-15.2f %-15.2f\n', values(i), probabilities(i), contributions(i), cumulative_sums(i));
end
fprintf('-----------------------------------------------------------------------------\n');
fprintf('Total Expected Value: %.2f\n', expected_value);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define parameters for a normal distribution
mu = 0; % Mean
sigma = 1; % Standard deviation

% Define the range of x values for plotting
x = linspace(mu - 4*sigma, mu + 4*sigma, 1000);

% Compute the probability density function
pdf_values = normpdf(x, mu, sigma);

% Create figure
figure;

% Plot the PDF
plot(x, pdf_values, 'LineWidth', 2);
title('Normal Distribution (PDF)');
xlabel('x');
ylabel('Density');

hold on;

% Fill area under the curve to visually indicate contribution to the expected value
area(x, pdf_values, 'FaceColor', 'cyan', 'EdgeColor', 'none');

% Highlight the mean
line([mu mu], [0 max(pdf_values)], 'Color', 'red', 'LineWidth', 2, 'LineStyle', '--');
text(mu, max(pdf_values)*0.9, 'Mean (Expected Value)', 'HorizontalAlignment', 'center', 'Color', 'red', 'FontSize', 12);

% Calculate expected value numerically (for demonstration)
expected_value = sum(x .* pdf_values) * (x(2) - x(1));

% Annotate the expected value calculation
text(mu - 2*sigma, max(pdf_values)*0.8, sprintf('Expected Value ≈ %.2f', expected_value), 'Color', 'blue', 'FontSize', 12);

hold off;

% Enhance visibility
axis tight;
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for the distributions
mu = 10; % Mean for both distributions
sigma = 3; % Standard deviation for the normal distribution

% Discrete distribution: Poisson
values = 0:20; % Range of discrete values
probabilities = poisspdf(values, mu); % Poisson probability mass function
contributions = values .* probabilities; % Contributions to expected value
cumulative_contributions = cumsum(contributions); % Cumulative sum of contributions

% Continuous distribution: Normal
x = linspace(mu - 4*sigma, mu + 4*sigma, 1000); % Range for continuous distribution
pdf_values = normpdf(x, mu, sigma); % Normal probability density function

% Calculate expected value from contributions
expected_value = sum(contributions);

% Create a figure
figure;

% Plot the discrete distribution using a bar chart
bar(values, probabilities, 'FaceColor', 'b', 'EdgeColor', 'b', 'FaceAlpha', 0.5);
hold on;

% Plot the continuous distribution using a line plot
plot(x, pdf_values, 'r', 'LineWidth', 2);

% Enhancements to the plot
title('Comparison of Discrete (Poisson) and Continuous (Normal) Distributions');
xlabel('Value');
ylabel('Probability/Density');
legend({'Poisson Distribution', 'Normal Distribution'}, 'Location', 'northwest');
grid on;

% Mark expected values
line([expected_value, expected_value], [0, max([probabilities, pdf_values])], 'Color', 'k', 'LineStyle', '--');
text(expected_value, max([probabilities, pdf_values])*0.95, sprintf('Expected Value = %.2f', expected_value), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 10, 'Color', 'k');

hold off;

% Print the table of probabilities, contributions, and cumulative contributions to the console
fprintf('\nValues, Probabilities, Contributions, and Cumulative Contributions:\n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('%-7s %-15s %-15s %-15s\n', 'Value', 'Probability', 'Contribution', 'Cum. Sum');
fprintf('--------------------------------------------------------------------------------\n');
for i = 1:length(values)
    fprintf('%-7d %-15.4f %-15.4f %-15.4f\n', values(i), probabilities(i), contributions(i), cumulative_contributions(i));
end
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Total Expected Value: %.2f\n', expected_value);