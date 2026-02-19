# This module implement various alternatives to the K-means clustering algorithm.
#
# Algorithms implemented:
# - Weighted K-means
# - Inverse Weighted K-means
# - Online K-means
# - Inverse Weighted Online K-means v1
# - Inverse Weighted Online K-means v2
# - Online K-Harmonic Means
# - Inverse Weighted Online K-means Topology Preserving Mapping
# - Inverse Weighted K-means Topology Preserving Mapping
#
# Notes:
# - All algorithms implemented in the Scikit-learn style, with fit, predict and transform methods.
#
# References:
# - BARBAKH, W., FYFE, C., ONLINE CLUSTERING ALGORITHMS 2008.

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class WeightedKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Weighted K-means clustering algorithm.

    Parameters:
        n_clusters (int): The number of clusters to form.
        max_iter (int): Maximum number of iterations of the k-means algorithm.
        tol (float): Relative tolerance with regards to inertia to declare convergence.

    Notes:
        (Algorithm description and references)
        We wish to form a performance function with the
        following properties:
            • Minimum performance gives an intuitively ‘good’
              clustering.
            • It creates a relationship between all data points
              and all prototypes.
        We have previously investigated the performance function
            J1 = \sum^N_{i=1} [ \sum^K_{j=1} ||x_i - m_j|| ] \min^K_{k=1} ||x_i - m_k||^2 (4)

        The rationale behind this performance function is
        that we wish to utilise the minimum distance in
        our learning algorithm but retain the global interaction which is necessary to ensure all prototypes
        play a part in the clustering. We have previously
        derived a batch clustering algorithm associated with
        this performance function by calculating the partial
        derivatives of (4) with respect to the prototypes, setting this to zero and hence finding the optimal prototypes’ positions. We call the resulting algorithm
        Weighted K-means (though recognising that other
        weighted versions of K-means have been developed
        in the literature). This gives a solution of
            m_r(t + 1) = [ \sum_{i \in V_r} x_i a_{ir} + \sum_{i \in V_j, j \neq r} x_i b_{ir} ] / [ \sum_{i \in V_r} a_{ir} + \sum_{i \in V_j, j \neq r} b_{ir} ] (5)
        where V_r contains the indices of data points that are
        closest to m_r, V_j contains the indices of all the other
        points and
        air = ||x_i − m_r(t)|| + 2 \sum^K_{j=1} ||x_i − m_j|| (6)
        bir = ||x_i − m_{k*}||^2 / || x_i - m_r(t) || (7)
        where again k∗ = arg mink x_i−m_k. We have given
        extensive analysis and simulations in Ref. 6 showing that this algorithm will cluster the data with
        the prototypes which are closest to the data points
        being positioned in such a way that the clusters can
        be identified. However, there can be some potential
        prototypes which are not sufficiently responsive to
        the data and so never move to identify a cluster. In
        fact, these points move to (a weighted) centre of the
        data set. This may be an advantage in some cases in
        that we can easily identify redundancy in the prototypes however it does waste computational resources
        unnecessarily.
    """
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        # Randomly initialize cluster centers
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        for iteration in range(self.max_iter):
            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.zeros_like(self.cluster_centers_)

            a_ir = distances + 2 * np.sum(distances, axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                min_distances = np.min(distances, axis=1)
                b_ir = np.where(distances != 0, (min_distances[:, np.newaxis] ** 2) / distances, 0)

            for r in range(self.n_clusters):
                V_r_indices = np.where(labels == r)[0]
                V_j_indices = np.where(labels != r)[0]

                sum_a = np.sum(a_ir[V_r_indices, r]) if V_r_indices.size > 0 else 0
                sum_b = np.sum(b_ir[V_j_indices, r]) if V_j_indices.size > 0 else 0

                numerator = np.sum(X[V_r_indices] * a_ir[V_r_indices, r][:, np.newaxis], axis=0) if V_r_indices.size > 0 else 0
                numerator += np.sum(X[V_j_indices] * b_ir[V_j_indices, r][:, np.newaxis], axis=0) if V_j_indices.size > 0 else 0

                denominator = sum_a + sum_b

                if denominator > 0:
                    new_centers[r] = numerator / denominator
                else:
                    new_centers[r] = self.cluster_centers_[r]

            # Check for convergence
            center_shift = np.linalg.norm(new_centers - self.cluster_centers_)
            if center_shift < self.tol:
                break
            self.cluster_centers_ = new_centers
        return self
    def transform(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return distances
    def score(self, X, y=None):
        distances = self.transform(X)
        closest_distances = np.min(distances, axis=1)
        return -np.sum(closest_distances ** 2)
    def predict(self, X):
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

class InverseWeightedKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Inverse Weighted K-means clustering algorithm.

    Parameters:
        n_clusters (int): The number of clusters to form.
        max_iter (int): Maximum number of iterations of the k-means algorithm.
        tol (float): Relative tolerance with regards to inertia to declare convergence.
        n (float): Exponent parameter n.
        p (float): Exponent parameter p.

    Notes:
        (Algorithm description and references)
    We also considered the performance function
    J2 = \sum^N_{i=1} [ \sum^K_{j=1} 1/||x_i - m_j||^{-p} ] \min^K_{k=1} ||x_i - m_k||^n (8)
    Using the same method as above, this gives the batch
    algorithm
    m_r(t + 1) = [ \sum_{i \in V_r} x_i a_{ir} + \sum_{i \in V_j, j \neq r} x_i b_{ir} ] / [ \sum_{i \in V_r} a_{ir} + \sum_{i \in V_j, j \neq r} b_{ir} ] (9)
    where V_r contains the indices of data points that are
    closest to m_r, V_j contains the indices of all the other
    points and
    a_ir = -(n - p) ||x_i - m_r(t)||^{n - p -2} - n ||x_i - m_r(t)||^{n - 2} \sum{j \neq k*} 1/||x_i - m_j||^p (10)
    b_ir = p ||x_i - m_{k*}||^n / || x_i - m_r(t) ||^{p + 2} (11)
    From the above, we see that n ≥ p if the direction
    of the first term is to be correct and n ≤ p + 2 to
    ensure stability in all parts of that equation. In practice, we have found that a viable algorithm may be
    found by using (11) for all prototypes (and thus never
    using (10) for the closest prototype). We call this the
    Inverse Weighted K-means Algorithm (IWK).
    """
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, n=2, p=2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n = n
        self.p = p
        self.cluster_centers_ = None
    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        # Randomly initialize cluster centers
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        for iteration in range(self.max_iter):
            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.zeros_like(self.cluster_centers_)

            with np.errstate(divide='ignore', invalid='ignore'):
                min_distances = np.min(distances, axis=1)
                a_ir_first_term = - (self.n - self.p) * np.power(distances, self.n - self.p - 2)
                sum_other_terms = np.sum(np.where(distances != min_distances[:, np.newaxis], 1.0 / np.power(distances, self.p), 0), axis=1)
                a_ir_second_term = - self.n * np.power(distances, self.n - 2) * sum_other_terms[:, np.newaxis]
                a_ir = a_ir_first_term + a_ir_second_term
                b_ir = np.where(distances != 0, self.p * np.power(min_distances[:, np.newaxis], self.n) / np.power(distances, self.p + 2), 0)

            for r in range(self.n_clusters):
                V_r_indices = np.where(labels == r)[0]
                V_j_indices = np.where(labels != r)[0]

                sum_a = np.sum(a_ir[V_r_indices, r]) if V_r_indices.size > 0 else 0
                sum_b = np.sum(b_ir[V_j_indices, r]) if V_j_indices.size > 0 else 0

                numerator = np.sum(X[V_r_indices] * a_ir[V_r_indices, r][:, np.newaxis], axis=0) if V_r_indices.size > 0 else 0
                numerator += np.sum(X[V_j_indices] * b_ir[V_j_indices, r][:, np.newaxis], axis=0) if V_j_indices.size > 0 else 0
                denominator = sum_a + sum_b

                if denominator > 0:
                    new_centers[r] = numerator / denominator
                else:
                    new_centers[r] = self.cluster_centers_[r]

            # Check for convergence
            center_shift = np.linalg.norm(new_centers - self.cluster_centers_)
            if center_shift < self.tol:
                break
            self.cluster_centers_ = new_centers
        return self
    def transform(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return distances
    def score(self, X, y=None):
        distances = self.transform(X)
        closest_distances = np.min(distances, axis=1)
        return -np.sum(closest_distances ** 2)
    def predict(self, X):
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

class OnlineKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Online K-means clustering algorithm.

    Parameters:
        n_clusters (int): The number of clusters to form.
        learning_rate (float): Learning rate for updating cluster centers.
        n_iter (int): Number of iterations over the dataset.
    Notes:
        (Algorithm description and references)
        The performance function for K-means may be written as
        J1 = \sum^N_{i=1} \min^K_{j=1} ||x_i - m_j||^2 (12)
        The implementation of the online K-means algorithm
        is as follows:
        (1) Initialization: initialize the cluster prototype vectors m_1,..., m_k
        (2) Loop for M iterations:
            (a) for each data vector x_i, set
                k∗ = argmin^K_{k=1} ||x_i − mk||
            (b) update the prototype m_{k∗} as
                m^{(new)}_{k*} = m_{k*} - ζ ∂J1/∂m_{k*}
                = m_{k*} + ζ (x_i - m_{k*})
            where ζ is a learning rate usually set to be a
            small positive number (e.g., 0.05).
        The learning rate can also gradually decrease during
        the learning process.
    """
    def __init__(self, n_clusters=8, learning_rate=0.05, n_iter=10):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.cluster_centers_ = None

    def partial_fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)

        n_samples, n_features = X.shape
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - self.cluster_centers_, axis=1)
            k_star = np.argmin(distances)
            self.cluster_centers_[k_star] += self.learning_rate * (X[i] - self.cluster_centers_[k_star])
        return self

    def initialize_centers(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

    def fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)
        else:
            raise ValueError("Centers already initialized. Use partial_fit for incremental updates.")
        for _ in range(self.n_iter):
            self.partial_fit(X, y)
        return self

    def transform(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return distances
    def score(self, X, y=None):
        distances = self.transform(X)
        closest_distances = np.min(distances, axis=1)
        return -np.sum(closest_distances ** 2)
    def predict(self, X):
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

class InverseWeightedOnlineKMeansV1(BaseEstimator, ClusterMixin, TransformerMixin):
    """Inverse Weighted Online K-means v1 clustering algorithm.

    Parameters:
        n_clusters (int): The number of clusters to form.
        learning_rate (float): Learning rate for updating cluster centers.
        n_iter (int): Number of iterations over the dataset.
        n (float): Exponent parameter n.
        p (float): Exponent parameter p.
    Notes:
        (Algorithm description and references)
        As shown in (9), in batch mode we have:
        mk = a_1x_1 + ··· + a_N x_N / a_1 + ··· + a_N (13)
        where
        if m_k is closest to x_i
            a_i = − (n − p) ||x_i − m_k||^{n − p −2} − n ||x_i − m_k||^{n − 2} \sum_{j \neq k*} 1/||x_i − m_j||^p
        otherwise
            a_i = p ||x_i − m_{k*}||^n / || x_i − m_k ||^{p + 2}
        mk∗ is the closest prototype to xi.

        In online mode, for IWK we can do something similar
        to (13) (taking into account that we receive one input
        sample at a time) as follows:
        (1) Initialization:
            — initialize the cluster prototype vectors
            m_1,..., m_K
            — initialize one dimensional vector, v, with K
            elements to one, v_1 = 1,...,v_K = 1
            note: v_k will represent the value that should be
            in denominator of (13) after feeding the input
            sample (it is used also for normalization).
        (2) Loop for M iterations:
            — for each data vector x_i, set
            k∗ = argmin^K_{k=1} ||x_i − m_k||
            and update all the prototypes m_k as
                m^{(new)}_k = m_k v_k + a_i x_i / v_k + a_i
                v^{(new)}_k = v_k + a_i
    """
    def __init__(self, n_clusters=8, learning_rate=0.05, n_iter=10, n=2, p=2):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n = n
        self.p = p
        self.cluster_centers_ = None
        self.v_ = None
    def partial_fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)
            self.v_ = np.ones(self.n_clusters)

        n_samples, n_features = X.shape
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - self.cluster_centers_, axis=1)
            k_star = np.argmin(distances)

            with np.errstate(divide='ignore', invalid='ignore'):
                min_distance = distances[k_star]
                # Compute a_i for closest cluster
                a_i_closest_first_term = - (self.n - self.p) * np.power(min_distance, self.n - self.p - 2)
                sum_other_terms = np.sum(np.where(distances != min_distance, 1.0 / np.power(distances, self.p), 0))
                a_i_closest_second_term = - self.n * np.power(min_distance, self.n - 2) * sum_other_terms

                # Compute a_i for other clusters
                a_i_other_terms = np.where(distances != 0, self.p * np.power(min_distance, self.n) / np.power(distances, self.p + 2), 0)

                # Combine to get a_i for all clusters
                a_i = a_i_other_terms
                a_i[k_star] = a_i_closest_first_term + a_i_closest_second_term


            self.cluster_centers_ = (self.cluster_centers_ * self.v_[:, np.newaxis] + a_i[:, np.newaxis] * X[i]) / (self.v_[:, np.newaxis] + a_i[:, np.newaxis])
            self.v_ += a_i
        return self
    def initialize_centers(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]
    def fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)
        else:
            raise ValueError("Centers already initialized. Use partial_fit for incremental updates.")
        return self.partial_fit(X, y)

    def transform(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return distances
    def score(self, X, y=None):
        distances = self.transform(X)
        closest_distances = np.min(distances, axis=1)
        return -np.sum(closest_distances ** 2)
    def predict(self, X):
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

class InverseWeightedOnlineKMeansV2(BaseEstimator, ClusterMixin, TransformerMixin):
    """Inverse Weighted Online K-means v2 clustering algorithm.

    Parameters:
        n_clusters (int): The number of clusters to form.
        learning_rate (float): Learning rate for updating cluster centers.
        n_iter (int): Number of iterations over the dataset.
        n (float): Exponent parameter n.
        p (float): Exponent parameter p.
    Notes:
        (Algorithm description and references)
    In this section we show how it is possible to allow all
    the units (prototypes) to learn, not only the winner
    as in K-means or the winner with its neighbors as in
    SOM. In this algorithm, it is not necessary to specify
    any functions for the neighbors as all units learn with
    every input sample.
    (1) Initialization: initialize the cluster prototype vectors m_1,..., m_k
    (2) Loop for M iterations:
    — for each data vector x_i, set
    k∗ = argmin^K_{k=1} ||x_i − m_k||
    and update all the prototypes m_k as
        m^{(new)}_{k*} = m_{k*} - ζ a_{ik*} (x_i - m_{k*} )
    where
        aik∗ = -(n + 1) ||x_i - m_{k*}||^{n - 1} - n ||x_i - m_{k*}||^{n - 2} \sum_{j \neq k*} ||x_i − m_j||

        m^{(new)}_k = m_ k - ζ (-||x_i - m_{k*}||^n / ||x_i - m_k||) (x_i - m_k )
    where ζ is a learning rate usually set to be a
    small positive number (e.g., 0.05).
    """
    def __init__(self, n_clusters=8, learning_rate=0.05, n_iter=10, n=2, p=2):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n = n
        self.p = p
        self.cluster_centers_ = None
    def partial_fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)

        n_samples, n_features = X.shape
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - self.cluster_centers_, axis=1)
            k_star = np.argmin(distances)
            min_distance = distances[k_star]

            # Compute a_ik* for closest cluster
            a_ik_star = - (self.n + 1) * np.power(min_distance, self.n - 1)
            sum_other_terms = np.sum(np.where(distances != min_distance, distances, 0))
            a_ik_star += - self.n * np.power(min_distance, self.n - 2) * sum_other_terms

            # Update closest cluster center
            self.cluster_centers_[k_star] -= self.learning_rate * a_ik_star * (X[i] - self.cluster_centers_[k_star])

            # Update other cluster centers
            for k in range(self.n_clusters):
                if k != k_star:
                    a_ik = - (min_distance ** self.n) / distances[k] if distances[k] != 0 else 0
                    self.cluster_centers_[k] -= self.learning_rate * a_ik * (X[i] - self.cluster_centers_[k])
        return self
    def initialize_centers(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]
    def fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)
        else:
            raise ValueError("Centers already initialized. Use partial_fit for incremental updates.")
        return self.partial_fit(X, y)
    def transform(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return distances
    def score(self, X, y=None):
        distances = self.transform(X)
        closest_distances = np.min(distances, axis=1)
        return -np.sum(closest_distances ** 2)
    def predict(self, X):
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

class KHarmonicMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Online K-Harmonic Means clustering algorithm.

    Parameters:
        n_clusters (int): The number of clusters to form.
        learning_rate (float): Learning rate for updating cluster centers.
        n_iter (int): Number of iterations over the dataset.
    Notes:
        (Algorithm description and references)
        In K-Harmonic means we have the following performance function:
            JHA = \sum^N_{i=1} K / [ \sum^K_{j=1} 1/||x_i - m_j||^2 ] (14)
        Then we wish to move the prototypes in such a
        way to minimize the performance function and hence
        identify the clusters
            ∂JHA/∂m_k = -K \sum^N_{i=1} 2 (x_i - m_k) / [ ||x_i - m_k||^4 [ \sum^K_{l=1} 1/||x_i - m_l||^2 ]^2 ] (15)
        Setting this equal to 0 and “solving” for the mk’s,
        we get a recursive formula (Batch Mode)
            m^{(new)}_k = \sum^N_{i=1} 1/ [d_{i,k}^4 [ \sum^K_{l=1} 1/d_{i,l}^2 ]^2] x_i / \sum^N_{i=1} 1/[ d_{i,k}^4 [ \sum^K_{l=1} 1/d_{i,l}^2 ]^2 ] (16)
            m^{(new)}_k = \sum^N_{i=1} 1/ [||x_i - m_k||^4 [ \sum^K_{l=1} 1/||x_i - m_l||^2 ]^2] x_i / \sum^N_{i=1} 1/[ ||x_i - m_k||^4 [ \sum^K_{l=1} 1/||x_i - m_l||^2 ]^2 ] (16)
        where we have used di,k for xi − mk to simplify
        the notation. There are some practical issues to deal
        with in the implementation details of which are given
        in Refs. 19, 20.
        Reference 20 have extensive simulations showing
        that this algorithm converges to a better solution
        (less prone to finding a local minimum because of
        poor initialization) than both standard K-means or
        a mixture of experts trained using the EM algorithm.
        3.4.1. Implementation (Online mode)
        From (15) we have:
            ∂JHA/∂m_k = -K 2 (x_i - m_k) / [ ||x_i - m_k||^4 [ \sum^K_{l=1} 1/||x_i - m_l||^2 ]^2 ] (17)

            m^{(new)}_k = m_k + ζ 2 K (x_i - m_k) / [ ||x_i - m_k||^4 [ \sum^K_{l=1} 1/||x_i - m_l||^2 ]^2 ] (18)
        where ζ is a learning rate usually set to be a small
        positive number (e.g., 0.05).
    """
    def __init__(self, n_clusters=8, learning_rate=0.05, n_iter=10):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.cluster_centers_ = None
    def partial_fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)

        n_samples, n_features = X.shape
        for i in range(n_samples):
            diff_with_clusters = X[i] - self.cluster_centers_
            distances = np.linalg.norm(diff_with_clusters, axis=1)
            sum_inv_distances_sq = np.sum(np.where(distances != 0, 1.0 / (distances ** 2), 0))
            norm = (distances ** 4 * (sum_inv_distances_sq ** 2))
            change = np.where(distances[:,None] != 0 | ~np.isfinite(norm)[:,None], - 2 * self.n_clusters * diff_with_clusters / norm[:, None], 0)
            self.cluster_centers_ -= self.learning_rate * change * (X[i] - self.cluster_centers_)
        return self
    def initialize_centers(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]
    def fit(self, X, y=None):
        if self.cluster_centers_ is None:
            self.initialize_centers(X)
        else:
            raise ValueError("Centers already initialized. Use partial_fit for incremental updates.")
        for _ in range(self.n_iter):
            self.partial_fit(X, y)
        return self
    def transform(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return distances
    def score(self, X, y=None):
        distances = self.transform(X)
        closest_distances = np.min(distances, axis=1)
        return -np.sum(closest_distances ** 2)
    def predict(self, X):
        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return labels

# Note: Topology Preserving Mapping versions not implemented yet.
#       Details of these algorithms can be found in the reference paper, Barbakh and Fyfe, Online Clustering Algorithms, 2008.
#       Implementation may take significant effort and testing, because implementation details are not fully specified in the paper.