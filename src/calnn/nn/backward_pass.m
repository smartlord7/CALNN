% Optimized Backpropagation with adaptive gradients (Adam optimizer)
function grads = backward_pass(net, X, real_distances, real_tag_position, n_anchors, activations, initial_anchors, lambda)
    num_layers = length(net.layers);
    grads = cell(num_layers, 1);

    % Precompute the predicted anchors and distances
    predicted_anchors = reshape(activations{end}, [n_anchors, 3]);
    diff_anchors = predicted_anchors - real_tag_position; % Precompute the difference
    predicted_distances = sqrt(sum(diff_anchors.^2, 2));  % Distance between predicted anchors and tag

    % Precompute 1/predicted_distances to avoid division during backprop
    inv_pred_distances = 1 ./ predicted_distances;  
    inv_pred_distances(isinf(inv_pred_distances)) = 0; % Handle possible div by zero cases

    % Gradient of distance loss w.r.t predicted distances (vectorized)
    dL_dPredDist = 2 * (predicted_distances - real_distances);

    % Gradient of distance loss w.r.t predicted anchor positions (vectorized)
    dPredDist_dPredAnchors = diff_anchors .* inv_pred_distances;  % Equivalent to (diff_anchors / predicted_distances)
    dL_dPredAnchors = dL_dPredDist .* dPredDist_dPredAnchors;      % Element-wise multiplication

    % Gradient of regularization term (element-wise operations for efficiency)
    reg_grad = 2 * lambda * (predicted_anchors - initial_anchors);

    % Total gradient w.r.t predicted anchor positions
    dL_dPredAnchors = dL_dPredAnchors + reg_grad;

    % Reshape gradient for backpropagation (in-place reshaping for efficiency)
    dL_dZ = reshape(dL_dPredAnchors, [], 1);  

    % Backpropagate through layers (vectorized where possible)
    for i = num_layers:-1:1
        if i == 1
            A_prev = X;  % Use input data for the first layer
        else
            A_prev = activations{i-1};  % Use the activations from the previous layer
        end

        % Efficient matrix multiplication for gradients w.r.t weights and biases
        grads{i}.dW = dL_dZ * A_prev';  % Matrix multiplication
        grads{i}.db = dL_dZ;            % Bias gradient (no need to reshape)

        % Backpropagate through the activation function
        if i > 1
            dL_dA_prev = net.layers{i}.W' * dL_dZ;  % Matrix multiplication for previous layer
            dZ_dA_prev = activation_derivative(net.activation_functions{i-1}, A_prev);  % Efficient activation derivative calculation
            dL_dZ = dL_dA_prev .* dZ_dA_prev;  % Element-wise product
        end
    end
end
