import numpy as np
from typing import Tuple, Optional

def get_lloyd_max_codebook(bits: int, seed: int = 42) -> np.ndarray:
    """
    Computes Lloyd-Max quantization codebook centroids for a standard normal distribution N(0, 1)
    using 1D K-means on a large sample.
    """
    n_levels = 2 ** bits
    np.random.seed(seed)
    # Generate standard normal samples
    samples = np.random.normal(0, 1, 200000)
    samples.sort()
    
    # Initialize centroids using percentiles of the distribution
    centroids = np.percentile(samples, np.linspace(100 / (2 * n_levels), 100 - 100 / (2 * n_levels), n_levels))
    
    # Run K-means clustering (typically converges in < 15 iterations)
    for _ in range(30):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        bins = np.digitize(samples, boundaries)
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            bin_samples = samples[bins == i]
            if len(bin_samples) > 0:
                new_centroids[i] = bin_samples.mean()
            else:
                new_centroids[i] = centroids[i]
        centroids = new_centroids
    return centroids

class TurboQuantex:
    """
    TurboQuantex algorithm for compressing high-dimensional embeddings:
    1. Random Rotation (PolarQuant preconditioning)
    2. Optimal Lloyd-Max Scalar Quantization (PolarQuant)
    3. Quantized Johnson-Lindenstrauss (QJL) 1-bit residual correction
    """
    def __init__(self, dim: int, bits: int = 2, use_qjl: bool = True, qjl_dim: int = 128, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.use_qjl = use_qjl
        self.qjl_dim = qjl_dim
        self.seed = seed
        
        # 1. Compute Lloyd-Max codebook
        self.centroids = get_lloyd_max_codebook(bits, seed)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0
        
        # 2. Generate stable random orthogonal matrix R
        np.random.seed(seed)
        A = np.random.normal(0, 1, (dim, dim))
        Q, _ = np.linalg.qr(A)
        self.R = Q
        
        # 3. Generate QJL random projection matrix S
        if use_qjl:
            # S is a random Gaussian matrix; entries are N(0, 1)
            self.S = np.random.normal(0, 1, (qjl_dim, dim))
            
    def compress(self, x: np.ndarray) -> Tuple[float, np.ndarray, Optional[np.ndarray], float]:
        """
        Compresses a vector x.
        
        Args:
            x: High-dimensional numpy array of shape (dim,)
            
        Returns:
            norm_x: Float representing the L2 norm of the original vector.
            indices: Integer array of shape (dim,) containing centroid indices.
            q_res: Boolean array of shape (qjl_dim,) containing sign bits (True = >=0, False = <0)
            norm_res: Float representing the L2 norm of the residual.
        """
        norm_x = float(np.linalg.norm(x))
        if norm_x < 1e-8:
            return 0.0, np.zeros(self.dim, dtype=np.int32), np.zeros(self.qjl_dim, dtype=bool) if self.use_qjl else None, 0.0
            
        x_norm = x / norm_x
        
        # Stage 1: PolarQuant random rotation
        y = np.dot(x_norm, self.R.T)
        
        # Scale coordinates to match N(0, 1) variance (since rotation distributes variance to 1/dim)
        y_std = y * np.sqrt(self.dim)
        
        # Map rotated coordinates to closest centroids in Lloyd-Max codebook
        indices = np.digitize(y_std, self.boundaries).astype(np.int32)
        indices = np.clip(indices, 0, len(self.centroids) - 1)
        y_std_quant = self.centroids[indices]
        
        # Decompress PolarQuant part to calculate residual
        y_quant = y_std_quant / np.sqrt(self.dim)
        x_norm_quant = np.dot(y_quant, self.R)
        
        # Calculate residual error in normalised space
        res = x_norm - x_norm_quant
        norm_res = float(np.linalg.norm(res))
        
        # Stage 2: QJL 1-bit residual correction
        q_res = None
        if self.use_qjl:
            # QJL sketch: sign of S * res
            proj_res = np.dot(self.S, res)
            q_res = (proj_res >= 0).astype(bool)
            
        return norm_x, indices, q_res, norm_res

    def decompress(self, norm_x: float, indices: np.ndarray) -> np.ndarray:
        """
        Reconstructs the vector using only the PolarQuant part (lossy reconstruction).
        """
        if norm_x < 1e-8:
            return np.zeros(self.dim)
            
        y_std_quant = self.centroids[indices]
        y_quant = y_std_quant / np.sqrt(self.dim)
        x_norm_quant = np.dot(y_quant, self.R)
        return x_norm_quant * norm_x

    def estimate_inner_product(self, norm_x: float, indices: np.ndarray, q_res: Optional[np.ndarray], norm_res: float, u: np.ndarray) -> float:
        """
        Estimates the inner product between the compressed vector and an uncompressed query u.
        
        Formula:
            <x, u> = <x_quant, u> + <res, u>
            where <res, u> is estimated using 1-bit QJL bits.
        """
        if norm_x < 1e-8:
            return 0.0
            
        # 1. PolarQuant contribution (reconstructed vector inner product)
        x_quant = self.decompress(norm_x, indices)
        ip_pq = float(np.dot(x_quant, u))
        
        if not self.use_qjl or q_res is None or norm_res < 1e-8:
            return ip_pq
            
        # 2. QJL residual correction
        # Map boolean array back to +1.0 and -1.0
        q_res_val = np.where(q_res, 1.0, -1.0)
        p_u = np.dot(self.S, u)
        
        # Unbiased estimator: ||x|| * ||res|| * sqrt(pi / 2) * mean(q_res * p_u)
        ip_qjl = norm_x * norm_res * np.sqrt(np.pi / 2.0) * float(np.mean(q_res_val * p_u))
        
        return ip_pq + ip_qjl
