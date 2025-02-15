% Forward pass with dynamic activation functions and residual connections
function [outputs, activations] = forward_pass(net, X)
    num_layers = length(net.layers);
    activations = cell(num_layers, 1);
    A = X; % Input layer activations

    for i = 1:num_layers
        Z = net.layers{i}.W * A + net.layers{i}.b;  % Linear combination
        % Apply the specified activation function for each layer
        A = apply_activation(net.activation_functions{i}, Z);

        activations{i} = A; % Store activations
    end

    outputs = A; % Final output (predicted anchor positions)
end