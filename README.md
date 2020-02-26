# Traj_Cluster
The package is to realize trajectory clustering given vehicles' travel history.          

It includes mainly five parts:          
* Distance functions to calculate the distance between 2D trajectories
* Slice long trajectories into short line segments by information theory
* Perform DBSCAN on line segments (so that line segments can be clustered based on their density and have flexible shapes.)
* Summarize one major trajectory for each cluster as the driver's typical travel trajectory
* Cluster trajs and perform colleborate filling with driver information.
