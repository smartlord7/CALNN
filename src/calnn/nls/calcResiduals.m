function residuals = calcResiduals(anchors, distances_noisy, tagPos, numAnchors, true_inter_anchor_distances)
    % Reshape the vector of anchor positions into matrix form
    anchors = reshape(anchors, [numAnchors, 3]);

    % Compute residuals between anchors and the tag
    residuals = zeros(numAnchors, 1);
    for i = 1:numAnchors
        predictedDistance = norm(anchors(i, :) - tagPos);
        residuals(i) = (predictedDistance - distances_noisy(i))^2;
    end

    % If true inter-anchor distances are provided, calculate inter-anchor residuals
    if ~isempty(true_inter_anchor_distances)
        % Find unique pairs of anchors for calculating inter-anchor distances
        [i_idx, j_idx] = find(triu(ones(numAnchors), 1)); % Index pairs for upper triangle
        
        % Compute the predicted inter-anchor distances
        predicted_interanchor_distances = sqrt(sum((anchors(i_idx, :) - anchors(j_idx, :)).^2, 2));
        
        % Calculate the residuals for inter-anchor distances
        inter_anchor_residuals = (predicted_interanchor_distances - ...
                                  true_inter_anchor_distances(sub2ind(size(true_inter_anchor_distances), i_idx, j_idx))).^2;
        
        % Append the inter-anchor residuals to the main residuals vector
        residuals = [residuals; inter_anchor_residuals];
    end

    % Flatten residuals into a single vector
    residuals = residuals(:);
end
