import numpy as np

# Thought process:
# 1. define the initial number of k clusters to split the data into
# 2. select k random points to serve as the initial centroids
# 3. calculate the Euclidean distance between the centroids and other points
# 4. assign the points to the closest centroid
# 5. calculate the centroid of each cluster
# 6. repeat steps 3-5

def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().replace("(", "").replace(")", "")
            # Split and parse each line into x, y coordinates
            x, y = map(float, line.split(","))
            points.append((x, y))  # Add point as a tuple
    return np.array(points)

def initial_centroids(points, cluster_no):
    # Randomly select initial centroids from points
    random_seed = np.random.choice(len(points), size=cluster_no, replace=False)
    initial_centroids = points[random_seed]
    
    # Set an initial radius as a fraction of the data range
    x_range = points[:, 0].max() - points[:, 0].min()  # Range of x values
    y_range = points[:, 1].max() - points[:, 1].min()  # Range of y values
    # Multiplying by 2 to scale the radius smaller
    initial_radius = min(x_range, y_range) / (2 * cluster_no) 

    return initial_centroids, initial_radius

def assign(points, centroids, radius):
    clusters = [[] for _ in centroids]
    outliers = list(map(tuple, points))  # Convert points to tuples for simpler comparison

    for point in points:
        assigned = False
        for i, centroid in enumerate(centroids):
            # Calculate the Euclidean distance from each point to the centroid
            distance = np.linalg.norm(point - centroid)
            if distance < radius:  # Check if within the radius to assign to that cluster
                clusters[i].append(point)
                point_tuple = tuple(point)
                if point_tuple in outliers:
                    outliers.remove(point_tuple)  # Remove from outliers if it's in the list
                assigned = True
                break  # Move to the next point after assigning
        if not assigned:
            # If the point cannot be assigned, it remains in outliers
            continue
    return clusters, outliers


def new_centroids(clusters, old_centroids):
    new_centroids = []
    for i, cluster in enumerate(clusters):
        if cluster:  # Avoid empty clusters
            new_centroid = np.mean(cluster, axis=0)
        else:
            # If a cluster is empty, retain the previous centroid
            new_centroid = old_centroids[i]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def adjust_radius(radius, clusters, outliers, points, max_outliers_limit=0.1):
    # Since picking initial centroids at random, we adjust the radius based on clustering performance
    if len(outliers) > max_outliers_limit * len(points):
        radius *= 1.1  # Increase radius by 10% if too many outliers
    else:
        radius *= 0.9  # Decrease radius by 10% if too few outliers
    return radius

def check_convergence(centroids, new_centroids, epsilon):
    # Calculate the distance moved for each centroid and check if below epsilon threshold
    shifts = np.linalg.norm(new_centroids - centroids, axis=1)
    return np.all(shifts < epsilon)

def k_means_with_adjusted_radius(points, n_clusters, max_iterations, epsilon):
    # Step 1: get initial centroids
    centroids, radius = initial_centroids(points, n_clusters)
    
    for iteration in range(max_iterations):
        # Step 2 & 3: calculate the Euclidean distance and assign points to clusters
        clusters, outliers = assign(points, centroids, radius)
        
        # Step 4: Calculate new centroids
        new_centroids_result = new_centroids(clusters, centroids)
        
        # Log the current iteration details
        print(f"Iteration {iteration + 1}")
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i + 1} - Centroid: {centroids[i]}")
            print(f"Points in Cluster: {cluster}")
        print("Current Outliers:", outliers)
        print("\n" + "-"*30)

        # Check for convergence
        if check_convergence(centroids, new_centroids_result, epsilon):
            print("Convergence reached.")
            break
        
        # Step 5: Adjust the radius based on clustering behavior
        radius = adjust_radius(radius, clusters, outliers, points)
        
        # Update centroids for the next iteration
        centroids = new_centroids_result
    
    # Final output
    print("Final Clusters and Centroids:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} - Final Centroid: {centroids[i]}")
        print(f"Points in Cluster: {cluster}")
    print("Outliers:", outliers)

    
# Read points from the file
file_path = "points_data.txt"
points = read_points(file_path)

# Define test parameters
n_clusters = 3
max_iterations = 10
epsilon = 0.01

k_means_with_adjusted_radius(points, n_clusters, max_iterations, epsilon)
