% Custom loss: Distance error with regularization using Huber Loss
function loss = compute_loss(predicted_anchors, real_distances, real_tag_position, n_anchors, initial_anchors, lambda, stds, delta)
    predicted_anchors = reshape(predicted_anchors, [n_anchors, 3]);

    % Compute predicted distances from each anchor to the tag
    predicted_distances = sqrt(sum((predicted_anchors - real_tag_position).^2, 2));

    % Use Huber loss instead of MSE for robustness to outliers
    distance_loss = huber_loss(predicted_distances, real_distances, delta);

    % Regularization term: Penalize deviation from initial anchors
    regularization_term = sum(sum(((predicted_anchors - initial_anchors).^2) ./ (stds.^2))); % Scaled by std deviation

    % Total loss is distance error + regularization term (with a weighting factor lambda)
    loss = distance_loss + lambda * regularization_term;
end