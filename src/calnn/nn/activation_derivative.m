% Function to compute the derivative of activation functions
function dZ_dA_prev = activation_derivative(activation_function, A)
    switch activation_function
        case 'relu'
            dZ_dA_prev = A > 0;
        case 'sigmoid'
            dZ_dA_prev = A .* (1 - A); % sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        case 'tanh'
            dZ_dA_prev = 1 - A.^2; % tanh'(x) = 1 - tanh(x)^2
        case 'linear'
            dZ_dA_prev = ones(size(A)); % Derivative of linear is 1
        otherwise
            error('Unsupported activation function.');
    end
end