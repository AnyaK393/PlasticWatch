import numpy as np

from sklearn.cluster import DBSCAN


class DBSCANClusteringEngine:

    def __init__(
        self,
        eps=0.15,
        min_samples=3
    ):
        """
        eps:
            Maximum neighborhood distance.

        min_samples:
            Minimum nearby points
            needed to form a cluster.
        """

        self.eps = eps
        self.min_samples = min_samples

    # ---------------------------------------------------
    # Main Clustering Method
    # ---------------------------------------------------

    def cluster_coordinates(
        self,
        coordinates
    ):
        """
        coordinates:
            List of [lat, lon] points
        """

        if len(coordinates) == 0:
            return []

        coordinates = np.array(coordinates)

        db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(coordinates)

        labels = db.labels_

        unique_clusters = set(labels)

        # Remove noise label
        unique_clusters.discard(-1)

        cluster_results = []

        for cluster_id in unique_clusters:

            cluster_points = coordinates[
                labels == cluster_id
            ]

            center_lat, center_lon = np.mean(
                cluster_points,
                axis=0
            )

            cluster_results.append({

                "cluster_id": int(cluster_id),

                "lat": float(center_lat),

                "lon": float(center_lon),

                "points_in_cluster": len(cluster_points)

            })

        return cluster_results