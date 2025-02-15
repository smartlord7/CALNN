% Define Estimator function (unchanged)s
function estimated_anchors = estimateAnchorPositions(n_anchors, initial_anchors, tagDistance, tag_position, tag_positions, tag_distances, true_inter_anchor_distances, bounds)

   estimatedAnchors = callibrate(n_anchors, initial_anchors, tagDistance, true_inter_anchor_distances, tag_position, 1, bounds, tag_distances, tag_positions, true);
    estimated_anchors = reshape(estimatedAnchors, [n_anchors, 3]);
end

% Example setup and initialization
rng(0);
n_anchors = 16;
std = 3; 
mx = [100; 100; 100];
b = [0, 100; 0, 100; 0, 100];
bounds = buildBounds(b, n_anchors);

real_anchors = generateAnchors(mx, n_anchors);
initial_anchors = real_anchors + std * randn(n_anchors, 3);


% Precompute true inter-anchor distances
true_inter_anchor_distances = zeros(n_anchors, n_anchors);
for i = 1:n_anchors-1
    for j = i+1:n_anchors
        true_inter_anchor_distances(i, j) = norm(real_anchors(i, :) - real_anchors(j, :));
    end
end

% Define 2D grid space for the tag position
[x_vals, y_vals] = meshgrid(linspace(0, mx(1), 20), linspace(0, mx(2), 20));
tag_positions = [];
tag_distances = [];
error_grid = zeros(size(x_vals));

% Loop through each grid position, calculate error at each
for ix = 1:numel(x_vals)
    disp("Position " + ix);
    % Current tag position in 2D
    tag_position = [x_vals(ix), y_vals(ix), 0];  % Fixed z-coordinate
    tag_positions = [tag_positions; tag_position];

    % True distances from current tag position to anchors
    tag_distance = sqrt(sum((real_anchors - tag_position).^2, 2))';
    tag_distances = [tag_distances; tag_distance];

    % Estimate anchor positions based on current tag position
    estimated_anchors = estimateAnchorPositions(n_anchors, initial_anchors, tag_distance', tag_position, tag_positions, tag_distances, true_inter_anchor_distances, bounds);
    initial_anchors = initial_anchors;

    % Calculate mean squared error between true and estimated anchors
    error = mean(sqrt(sum((estimated_anchors - real_anchors).^2, 2)));
    display("MSE: " + error);
    error_grid(ix) = log(error);
end

% Plotting the error as a 3D surface
figure;
surf(x_vals, y_vals, error_grid, 'EdgeColor', 'none');
colormap(jet); % Color map for error intensity
colorbar; % Display color scale
xlabel('X');
ylabel('Y');
zlabel('Estimation Error');
title('Anchor Position Estimation Error Across X-Y Space');
