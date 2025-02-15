% Function to apply activation functions
function A = apply_activation(activation_function, Z)
    switch activation_function
        case 'relu'
            A = max(0, Z);
        case 'sigmoid'
            A = 1 ./ (1 + exp(-Z));
        case 'tanh'
            A = tanh(Z);
        case 'linear'
            A = Z;
        otherwise
            error('Unsupported activation function.');
    end
end