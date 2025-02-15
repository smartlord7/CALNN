function residuals = calcWeightedResiduals(anchors, distances_noisy, tagPos, numAnchors, numSamples, interanchor_distances)
    % Reshape the flat vector of anchors back into matrix form
    anchors = reshape(anchors, [numAnchors, 3]);
    
    % Define weights for each sample (linear decay from 1 to 0.1)
    weights = linspace(1, 0.1, numSamples)';
    
    % Expand tag positions and anchors for vectorized distance computation
    % tagPos_exp: numSamples x 1 x 3, anchors_exp: 1 x numAnchors x 3
    tagPos_exp = reshape(tagPos, [numSamples, 1, 3]);
    anchors_exp = reshape(anchors, [1, numAnchors, 3]);
    
    % Calculate predicted distances for each tag position and anchor
    predictedDistances = sqrt(sum((anchors_exp - tagPos_exp).^2, 3)); % numSamples x numAnchors
    
    % Calculate residuals matrix with weights applied
    residualsMatrix = (predictedDistances - distances_noisy) .* weights; % element-wise weighting

    % Convert residuals matrix to a column vector
    residuals = residualsMatrix(:);
    
    % Check if inter-anchor distances are provided and not empty
    if ~isempty(interanchor_distances)
        % Precompute pairwise inter-anchor distances (vectorized)
        [i_idx, j_idx] = find(triu(ones(numAnchors), 1)); % Get index pairs for unique upper triangle
        
        % Calculate predicted inter-anchor distances
        predicted_interanchor_distances = sqrt(sum((anchors(i_idx, :) - anchors(j_idx, :)).^2, 2));
        
        % Calculate residuals for inter-anchor distances
        anchor_dist_errors = (predicted_interanchor_distances - ...
                              interanchor_distances(sub2ind(size(interanchor_distances), i_idx, j_idx))).^2;
        
        % Append inter-anchor residuals to the main residuals vector
        residuals = [residuals; anchor_dist_errors];
    end
end
