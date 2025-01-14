function [variogram, lagDistances] = calculateSemivariogram(data, nrbins)
    % Calculate pairwise distances and indices of pairs
    distances = pdist(data);  % vector of pairwise distances
    distMatrix = squareform(distances); % Convert to a square matrix form
    
    maxDistance = max(distances);  % Maximum pairwise distance
    lagDistances = linspace(0, maxDistance, nrbins);
    variogram = zeros(1, length(lagDistances)-1);

    % Compute semivariance for each bin
    for i = 1:length(lagDistances)-1
        minDist = lagDistances(i);
        maxDist = lagDistances(i+1);

        % Logical array for distances within the current bin
        binIndices = (distMatrix > minDist) & (distMatrix <= maxDist);
        
        % Extract pairs that match the bin condition
        pairs = [];
        for row = 1:size(data,1)
            for col = row+1:size(data,1)  % Only upper triangle needed
                if binIndices(row, col)
                    pairs = [pairs; data(row,:), data(col,:)];
                end
            end
        end

        % Calculate semivariance
        if ~isempty(pairs)
            diffs = pairs(:, 1:2:end) - pairs(:, 2:2:end);
            variogram(i) = mean(diffs(:).^2) / 2;
        end
    end
end