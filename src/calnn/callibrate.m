function final_anchors = callibrate(n_anchors, initial_anchors, real_distances, true_inter_anchor_distances, real_tag_position, std, bounds, noisyDistancesHistory, tagPositions, isDynamic)
    % CALLIBRATE: Estimates anchor positions based on measured distances.
    % 
    % This function applies a neural network-based optimization technique to
    % refine anchor positions given noisy distance measurements. The algorithm
    % first uses an artificial neural network to generate an initial estimation,
    % followed by a nonlinear least squares optimization step for further refinement.
    %
    % This function should be applied at well-known tag trajectory points, such as
    % the corners of a predefined rectangular trajectory. The history of tag
    % positions and anchor distances should be stored and updated at each of these
    % key points. This ensures better convergence and reduces calibration error
    % over time.
    %
    % PARAMETERS:
    %   n_anchors (integer):
    %       Number of anchors in the environment.
    %   initial_anchors (n_anchors x 3 matrix):
    %       Initial estimated positions of anchors.
    %   real_distances (1 x n_anchors vector):
    %       True distances between the tag and anchors (known at each trajectory point by time ToA measurements between anchors, subject to measurement noise. 
    %       Each current tag-anchors distances then makes part of noisyDistancesHistory at the next trajectory point).
    %   true_inter_anchor_distances (n_anchors x n_anchors matrix):
    %       Pairwise true distances between anchors (known a priori by ToA measurements between anchors, subject to measurement noise).
    %   real_tag_position (1 x 3 vector):
    %       Known tag position in 3D space, point that is contained in the trajectory, i.e. a corner of a rectangle.
    %   std (n_anchors x 3 matrix):
    %       Standard deviation matrix representing the uncertainty in the initial
    %       anchor position guesses for each coordinate (x, y, z).
    %   bounds (n_anchors x 6 matrix):
    %       Constraints on anchor positions ([x_min, x_max, y_min, y_max,
    %       z_min, z_max]). This is trivial to do, it is based on the room dimensions. 
    %   noisyDistancesHistory (m x n_anchors matrix):
    %       Historical noisy distance measurements, where m is the number of past samples (known at each trajectory point by ToA measurements between anchors, subject to measurement noise).
    %   tagPositions (m x 3 matrix):
    %       Historical tag positions, corresponding to the past m samples, known a priori since are contained in the predefined trajectory.
    %   isDynamic (boolean):
    %       Flag indicating if the calibration process considers past samples.
    %       If true, the calibration is dynamically enhanced using noisyDistancesHistory
    %       and tagPositions.
    %
    % RETURNS:
    %   final_anchors (n_anchors x 3 matrix):
    %       Estimated refined anchor positions.
    %
    % USAGE EXAMPLE:
    %   % Define environment parameters
    %   n_anchors = 16;
    %   max_coord = [100, 100, 100];
    %   std_dev = ones(n_anchors, 3) * 2; % Uncertainty of ±2 meters in all coordinates
    %   bounds = [zeros(n_anchors, 1), ones(n_anchors, 1) * max_coord(1), ...
    %            zeros(n_anchors, 1), ones(n_anchors, 1) * max_coord(2), ...
    %            zeros(n_anchors, 1), ones(n_anchors, 1) * max_coord(3)];
    %
    %   % Generate true anchor positions and add noise to simulate initial estimates
    %   real_anchors = rand(n_anchors, 3) .* max_coord;
    %   initial_anchors = real_anchors + std_dev .* randn(n_anchors, 3); %Initial anchors guesses, i.e. rough estimates of the actual anchor positions
    %
    %   % Compute true inter-anchor distances
    %   true_inter_anchor_distances = squareform(pdist(real_anchors)); % Known a priori, by measuring the distances between every possible pair of anchors
    %
    %   % Simulate a known trajectory and perform dynamic calibration at each point
    %   num_measurements = 10;
    %   trajectory_points = [linspace(10, 90, num_measurements)', linspace(10, 90, num_measurements)', zeros(num_measurements, 1)];
    %   final_anchors = initial_anchors;
    %
    %   for i = 1:num_measurements
    %       real_tag_position = trajectory_points(i, :);
    %       real_distances = sqrt(sum((real_anchors - real_tag_position).^2, 2))';
    %       noisyDistancesHistory = real_distances [
    %       randn(num_measurements, n_anchors); % Simulates measurement noise
    %       tagPositions = repmat(real_tag_position, num_measurements, 1) + randn(num_measurements, 3);
    %       final_anchors = callibrate(n_anchors, final_anchors, real_distances, ...
    %                                 true_inter_anchor_distances, real_tag_position, ...
    %                                 std_dev, bounds,
    %                                 noisyDistancesHistory, tagPositions,
    %                                 true); % The previous anchors
    %                                        % estimations are reused in the next
    %                                        % trajectory point, converging better
    %                                        % to the real anchor positions
    %   end
    %
    % NETWORK INITIALIZATION
    layer_sizes = [3 * n_anchors + 3, 10, 10, 3 * n_anchors];
    activation_functions = {'relu', 'sigmoid', 'linear'};
    net = initialize_network(layer_sizes, activation_functions);

    % PARAMETERS STRUCTURE
    params = struct();
    params.max_iters = 6000;  % Maximum iterations for optimization
    params.lambda = 0.025;    % Regularization parameter
    params.lr = 1e-2;         % Learning rate for optimization
    params.stds = std;        % Standard deviations for uncertainty modeling
    params.delta = 1;         % Huber loss delta parameter

    % PREPARE INPUT DATA
    input = initial_anchors(:);
    input = cat(1, input, real_tag_position');

    % RUN OPTIMIZATION USING ADAM
    [net, ~] = adam_optimization(net, input, real_distances, real_tag_position, n_anchors, initial_anchors, params);
    
    % FINAL ESTIMATED ANCHOR POSITIONS
    [final_outputs, ~] = forward_pass(net, input);
    final_anchors = reshape(final_outputs, [n_anchors, 3]);
    
    % NONLINEAR LEAST SQUARES REFINEMENT
    if isDynamic
        final_anchors = nonlinearLeastSquares(noisyDistancesHistory, true_inter_anchor_distances, final_anchors, tagPositions, bounds, isDynamic);
    else
        final_anchors = nonlinearLeastSquares(real_distances, true_inter_anchor_distances, final_anchors, tagPositions, bounds, isDynamic);
    end
end
