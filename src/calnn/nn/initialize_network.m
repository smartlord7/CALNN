% Initialize the network with modular architecture and residual connections
function net = initialize_network(layer_sizes, activation_functions)
    num_layers = length(layer_sizes) - 1;
    net.layers = cell(num_layers, 1);
    net.activation_functions = activation_functions; % Store the activation functions

    % Initialize weights and biases with small random values
    for i = 1:num_layers
        net.layers{i}.W = randn(layer_sizes(i+1), layer_sizes(i)) * 0.1; % Larger weights
        net.layers{i}.b = zeros(layer_sizes(i+1), 1);                    % Biases initialized to zero
    end
end
