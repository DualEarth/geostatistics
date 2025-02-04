function dice_rolls()
    rng(42); % Set random seed for reproducibility
    
    % Number of rolls
    maxRolls = 1000;
    
    % Prior beliefs: Dirichlet prior, assuming the die is fair initially
    alpha = ones(1, 6); % Uniform Dirichlet prior (similar to having seen each face once)
    
    % Simulating a potentially biased die
    true_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]; % Biased towards 6
    
    % To store the estimated expected values
    expected_values_frequentist = zeros(1, maxRolls);
    expected_values_bayesian = zeros(1, maxRolls);
    
    % Initial count/frequency of each face
    face_counts = zeros(1, 6);
    
    for roll = 1:maxRolls
        % Simulate a roll based on the true probabilities
        face = find(rand <= cumsum(true_probs), 1, 'first');
        
        % Update counts
        face_counts(face) = face_counts(face) + 1;
        
        % Frequentist estimate: simple average
        expected_values_frequentist(roll) = sum(face_counts .* (1:6)) / sum(face_counts);
        
        % Bayesian update: Update alpha for Dirichlet
        alpha(face) = alpha(face) + 1;
        
        % Calculate the expected value from the current posterior distribution
        posterior = alpha / sum(alpha);
        expected_values_bayesian(roll) = sum(posterior .* (1:6));
        
        % Display results every 100 rolls
        if mod(roll, 100) == 0
            fprintf('Roll %d: Frequentist EV = %.2f, Bayesian EV = %.2f\n', ...
                    roll, expected_values_frequentist(roll), expected_values_bayesian(roll));
        end
    end
    
    % Plot the convergence of expected values
    figure;
    plot(1:maxRolls, expected_values_frequentist, 'b', 'DisplayName', 'Frequentist');
    hold on;
    plot(1:maxRolls, expected_values_bayesian, 'r', 'DisplayName', 'Bayesian');
    xlabel('Number of Rolls');
    ylabel('Expected Value');
    title('Convergence of Expected Value Estimates');
    legend show;
    grid on;
end