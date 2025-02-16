# CALNN
Algorithm developed to calibrate anchors in a UWB indoor location system.


# How-to Guide for UWB Anchor Calibration using CALNN

A piece of advice: store all the data involved. This way, in the end of this process, we reduce drastically the possibility of having something amiss :-).



## Step 1: Review Documentation
Carefully read the documentation of the `calibrate()` function located in the `src/` directory to understand its purpose, parameters, and expected outputs. Test the code on generated use cases, such as the one presented in the documentation.

## Step 2: Retrieve True Anchor Positions
Obtain the true positions of the anchors with high precision for comparison against both our approach and the manufacturer's approach (e.g., using laser-based measurements). Repeat this process multiple times to minimize the impact of random errors.

## Step 3: Compute Inter-Anchor Distances
Measure and store inter-anchor distances in advance. Repeat this process several times to reduce noise effects and use the mean value for better accuracy.

## Step 4: Define a Calibration Trajectory
Design a trajectory for calibration with multiple known points. For example, if using a rectangular trajectory, the known points could be its corners. A higher number of points improves calibration accuracy. It is recommended to use at least eight points, with the ideal trajectory being an infinity (∞) symbol.

## Step 5: Execute the Trajectory
Begin moving the tag along the defined trajectory, ensuring it starts at a known point.

## Step 6: Run the Calibration Function
Execute the `calibrate()` function at each known point along the trajectory, using:
- The previously estimated anchor positions
- The tag's previous positions
- The measured tag-to-anchor distances

Store the intermediate anchor position estimates for further analysis and evaluation.

## Step 7: Retrieve Manufacturer's Anchor Estimates
Obtain the anchor position estimates generated by the manufacturer’s algorithm. If available, also collect the intermediate estimates of anchor positions for additional comparison.

## Step 8: Compare Calibration Results
At the end of the trajectory, compare the anchor positions estimated by the `CALNN` method against those from the baseline UWB anchor calibration algorithm.

## Step 9: Repeat for Statistical Relevance
Repeat steps 3–8 multiple times to ensure statistical reliability. Store all collected data properly for later analysis.

## Step 10: Evaluate Tag Positioning Accuracy
Define a separate trajectory with known intermediate points to assess how both the baseline and proposed estimators influence the accuracy of the tag’s location. To do this:
- Compute the mean positions of both sets of estimated anchors.
- Perform trilateration twice: once using the manufacturer's estimated anchor positions and once using the positions estimated by the proposed algorithm.
- Repeat multiple times for statistical significance and store all results, including the estimated tag positions from both trilateration executions.

By following these steps, you ensure a robust evaluation of anchor calibration and its impact on positioning accuracy.

