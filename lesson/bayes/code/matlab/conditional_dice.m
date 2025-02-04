% Total number of possible outcomes for two dice
totalOutcomes = 6 * 6;

% Event A: First die shows a number greater than 4 (i.e., 5 or 6)
A = [5 6];
% All possible outcomes for the second die
secondDie = 1:6;

% Generate all combinations where first die is greater than 4
% Each row in outcomesA corresponds to a possible outcome where Event A occurs
outcomesA = combvec(A, secondDie)';

% Number of favorable outcomes for Event A
numFavorableA = size(outcomesA, 1);

% Probability of Event A
P_A = numFavorableA / totalOutcomes;

% Event B given A: Sum is greater than 8
% Filter outcomesA to only those rows where the sum of the dice is greater than 8
outcomesBgivenA = outcomesA(sum(outcomesA, 2) > 8, :);

% Number of favorable outcomes for Event B given A
numFavorableBgivenA = size(outcomesBgivenA, 1);

% Probability of Event B given A
P_BgivenA = numFavorableBgivenA / numFavorableA;

fprintf('Probability of rolling more than 4 on the first die (P(A)): %.2f\n', P_A);
fprintf('Probability of rolling a sum greater than 8 given the first die is more than 4 (P(B|A)): %.2f\n', P_BgivenA);