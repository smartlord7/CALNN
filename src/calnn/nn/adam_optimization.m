% Optimized Adam optimization with random perturbation when loss stagnates
function [net, loss_history] = adam_optimization(net, X, real_distances, real_tag_position, n_anchors, initial_anchors, params)

    % Use default values if certain fields do not exist in params struct
    if ~isfield(params, 'max_iters'), params.max_iters = 5000; end
    if ~isfield(params, 'lambda'), params.lambda = 0.0025; end
    if ~isfield(params, 'lr'), params.lr = 1e-3; end
    if ~isfield(params, 'stds'), params.stds = ones(n_anchors, 3); end
    if ~isfield(params, 'delta'), params.delta = 1; end
    if ~isfield(params, 'plot_'), params.plot_ = false; end
    if ~isfield(params, 'real_anchors'), params.real_anchors = []; end

    % Adam hyperparameters, with default values if missing
    if ~isfield(params, 'beta1'), params.beta1 = 0.9; end
    if ~isfield(params, 'beta2'), params.beta2 = 0.999; end
    if ~isfield(params, 'epsilon'), params.epsilon = 1e-8; end
    if ~isfield(params, 'tol'), params.tol = 1e-6; end
    if ~isfield(params, 'stagnation_window'), params.stagnation_window = 100; end
    if ~isfield(params, 'stagnation_threshold'), params.stagnation_threshold = 1e-4; end
    if ~isfield(params, 'perturbation_strength'), params.perturbation_strength = 1; end
    if ~isfield(params, 'perturbation_decay_factor'), params.perturbation_decay_factor = 0.95; end

    % Initialize variables for Adam optimizer
    max_iters = params.max_iters;
    beta1 = params.beta1;
    beta2 = params.beta2;
    epsilon = params.epsilon;
    tol = params.tol;
    lambda = params.lambda;
    lr = params.lr;
    stds = params.stds;
    delta = params.delta;
    plot_ = params.plot_;
    real_anchors = params.real_anchors;

    % Stagnation detection
    stagnation_window = params.stagnation_window;
    stagnation_threshold = params.stagnation_threshold;
    perturbation_strength = params.perturbation_strength;
    perturbation_decay_factor = params.perturbation_decay_factor;

    % Initialize Adam parameters
    num_layers = length(net.layers);
    mW = cell(num_layers, 1);
    vW = cell(num_layers, 1);
    mb = cell(num_layers, 1);
    vb = cell(num_layers, 1);

    % Preallocate moment estimates
    for i = 1:num_layers
        mW{i} = zeros(size(net.layers{i}.W));
        vW{i} = zeros(size(net.layers{i}.W));
        mb{i} = zeros(size(net.layers{i}.b));
        vb{i} = zeros(size(net.layers{i}.b));
    end
    
    loss_history = zeros(max_iters, 1);  % Preallocate array for loss values
    if ~isempty(real_anchors)
        rmse_history = zeros(max_iters, 1);  % Preallocate array for RMSE values
    end


    % Main optimization loop
    for iter = 1:max_iters
        
        % Forward pass
        [outputs, activations] = forward_pass(net, X);
        
        % Compute loss
        loss = compute_loss(outputs, real_distances, real_tag_position, n_anchors, initial_anchors, lambda, stds, delta);
        %disp(['Iteration ', num2str(iter), ' Loss: ', num2str(loss)]);

        % Store the loss for this iteration
        loss_history(iter) = loss;

        if ~isempty(real_anchors)
            outputs_ = reshape(outputs, [n_anchors, 3]);
            rmse = root_mean_squared_error(outputs_, real_anchors);
            rmse_history(iter) = rmse;
        end

        % Check for stagnation
        if iter > stagnation_window
            loss_diff = abs(loss_history(iter) - loss_history(iter - stagnation_window));
            if mod(iter, stagnation_window) == 0
                % Dynamically adjust perturbation strength based on iteration
                perturbation_strength = perturbation_strength * perturbation_decay_factor;
                if loss_diff < stagnation_threshold && perturbation_strength > 1e-6
                    % Apply random perturbation to the weights and biases
                    perturbation = perturbation_strength * randn(num_layers, 1);
                    for i = 1:num_layers
                        net.layers{i}.W = net.layers{i}.W + perturbation(i) * randn(size(net.layers{i}.W));
                        net.layers{i}.b = net.layers{i}.b + perturbation(i) * randn(size(net.layers{i}.b));
                    end
                end
            end
        end

        % Compute gradients via backpropagation
        grads = backward_pass(net, X, real_distances, real_tag_position, n_anchors, activations, initial_anchors, lambda);

        % Adam optimization: Update weights and biases
        for i = 1:num_layers
            % Update moment estimates
            mW{i} = beta1 * mW{i} + (1 - beta1) * grads{i}.dW;
            vW{i} = beta2 * vW{i} + (1 - beta2) * grads{i}.dW.^2;
            mb{i} = beta1 * mb{i} + (1 - beta1) * grads{i}.db;
            vb{i} = beta2 * vb{i} + (1 - beta2) * grads{i}.db.^2;

            % Bias-corrected estimates
            mW_hat = mW{i} / (1 - beta1^iter);
            vW_hat = vW{i} / (1 - beta2^iter);
            mb_hat = mb{i} / (1 - beta1^iter);
            vb_hat = vb{i} / (1 - beta2^iter);

            % Update weights and biases
            net.layers{i}.W = net.layers{i}.W - lr * mW_hat ./ (sqrt(vW_hat) + epsilon);
            net.layers{i}.b = net.layers{i}.b - lr * mb_hat ./ (sqrt(vb_hat) + epsilon);
        end
        
        % Check for convergence
        if loss < tol
            loss_history = loss_history(1:iter);  % Trim loss history up to the current iteration
            if real_anchors ~= false
                rmse_history = rmse_history(1:iter);
            end
            break;
        end
    end

    % Trim loss history in case we converged early
    if iter < max_iters
        loss_history = loss_history(1:iter);
        if ~isempty(real_anchors)
            rmse_history = rmse_history(1:iter);
        end
    end

    % Generate plot at the end
    if plot_
        figure('Units', 'Normalized', 'OuterPosition', [0 0 1 1]);  % Create a fullscreen figure
        plot(1:length(loss_history), loss_history, 'LineWidth', 2);
        xlabel('Epoch');
        ylabel('Loss');
        title(['Loss over Epochs (LR: ', num2str(lr), ', Lambda: ', num2str(lambda), ')']);
        grid on;

        % Save the plot with a meaningful filename
        filename = ['loss_plot', '_lr', num2str(lr), '_lambda', num2str(lambda), '.png'];
        saveas(gcf, filename);
        disp(['Plot saved as ', filename]);

        if ~isempty(real_anchors)
            figure('Units', 'Normalized', 'OuterPosition', [0 0 1 1]);  % Create a fullscreen figure
            plot(1:length(rmse_history), rmse_history, 'LineWidth', 2);
            xlabel('Epoch');
            ylabel('Loss');
            title(['RMSE over Epochs (LR: ', num2str(lr), ', Lambda: ', num2str(lambda), ')']);
            grid on;
    
            % Save the plot with a meaningful filename
            filename = ['rmse_plot', '_lr', num2str(lr), '_lambda', num2str(lambda), '.png'];
            saveas(gcf, filename);
            disp(['Plot saved as ', filename]);
        end
    end
end